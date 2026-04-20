from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

with open('expenses_model.pkl', 'rb') as f:
    model = pickle.load(f)

expenses_kmn = model['kmn']
expenses_sc = model['sc']
expenses_cluster = model['cluster_profile']



def recommend_budget(user_data, kmn, sc, cluster_profile):
    try:
        column_mapping = {
            'Groceries': 'Groceries',
            'Transport': 'Transport', 
            'Eating Out': 'Eating_Out',
            'Entertainment': 'Entertainment',
            'Miscellaneous': 'Miscellaneous',
            'Desired Savings': 'Desired_Savings',
            'Disposable Income': 'Disposable_Income'
        }
        
        user_data_mapped = user_data.rename(columns=column_mapping)
        
        feature = user_data_mapped[['Groceries','Transport','Eating_Out',
                                   'Entertainment','Miscellaneous',
                                   'Desired_Savings','Disposable_Income']]
        
        scaled_x = sc.transform(feature)

        cluster_id = kmn.predict(scaled_x)[0]
        cluster_pro = cluster_profile.loc[cluster_id]

        user_saving = (
            user_data_mapped['Disposable_Income'].values[0] -
            user_data_mapped['Desired_Savings'].values[0]
        )

        if user_saving >= 0:
            return {
                "Message": f"Your saving goal is achievable. You are saving ₹{user_saving:.2f} monthly.",
                "Saving": user_saving,
                "Suggestions": []
            }
        
        shortfall = abs(user_saving)
        suggestions = []

        for col in ['Groceries','Transport','Eating_Out','Entertainment','Miscellaneous']:
            user_val = user_data_mapped[col].values[0]
            cluster_val = cluster_pro[col]

            if user_val > cluster_val:
                suggestions.append(
                    f"Reduce {col}: currently ₹{user_val:.2f}, suggested ₹{cluster_val:.2f}"
                )

        return {
            "Message": f"Your savings are not achievable. You must reduce your expenses by ₹{shortfall:.2f} to reach your saving goal.",
            "Suggestions": suggestions,
        }

    except Exception as e:
        return {"error": str(e)}


# Flask API (clean + complete)
@app.route("/recommend-budget", methods=["POST"])
def recommend_budget_endpoint():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        field_mapping = {
            'Groceries': 'Groceries',
            'Transport': 'Transport',
            'Eating Out': 'Eating_Out',
            'Entertainment': 'Entertainment',
            'Miscellaneous': 'Miscellaneous',
            'Desired Savings': 'Desired_Savings',
            'Disposable Income': 'Disposable_Income'
        }

        model_data = {}

        for frontend_name, backend_name in field_mapping.items():
            if frontend_name not in data:
                return jsonify({"error": f"Missing field: {frontend_name}"}), 400

            try:
                model_data[backend_name] = float(data[frontend_name])
            except:
                return jsonify({"error": f"Invalid value for {frontend_name}"}), 400

        df = pd.DataFrame([model_data])

        rec = recommend_budget(df, expenses_kmn, expenses_sc, expenses_cluster)

        return jsonify(rec)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Internal server error"}), 500


# Run server
if __name__ == "__main__":
    app.run(debug=True, port=8000)