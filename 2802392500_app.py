# Stevanus Gerald Marconus - 2802392500
import streamlit as st
import joblib
import pandas as pd

artifacts = joblib.load('artifacts/best_model.pkl')
model = artifacts['pipeline']
le_target = artifacts['le_target']


def main():
    st.set_page_config(page_title="Credit Score Predictor", layout="wide")
    st.title("Credit Score Prediction")
    st.write("Stevanus Gerald Marconus — 2802392500")

    with st.sidebar:
        st.header("Input Guide")
        st.write("Isi data nasabah di form, lalu klik **Predict** untuk melihat hasil prediksi Credit Score.")
        st.info("Model akan memprediksi apakah Credit Score nasabah termasuk **Good**, **Standard**, atau **Poor**.")

    st.subheader("Data Nasabah")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Informasi Pribadi**")
            age = st.number_input("Age", min_value=18, max_value=100, step=1, value=30)
            occupation = st.selectbox("Occupation", [
                "Unknown", "Accountant", "Architect", "Developer", "Doctor",
                "Engineer", "Entrepreneur", "Journalist", "Lawyer",
                "Manager", "Media_Manager", "Mechanic", "Musician",
                "Scientist", "Teacher", "Writer"
            ])
            annual_income = st.number_input("Annual Income", min_value=0.0, step=100.0, value=50000.0)
            monthly_inhand_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, step=100.0, value=4000.0)
            num_bank_accounts = st.number_input("Num Bank Accounts", min_value=0, max_value=20, step=1, value=3)
            num_credit_card = st.number_input("Num Credit Card", min_value=0, max_value=20, step=1, value=2)
            interest_rate = st.number_input("Interest Rate (%)", min_value=0, max_value=50, step=1, value=10)
            num_of_loan = st.number_input("Num of Loan", min_value=0, max_value=20, step=1, value=2)

        with col2:
            st.markdown("**Riwayat Kredit**")
            delay_from_due_date = st.number_input("Delay from Due Date (days)", min_value=0, max_value=100, step=1, value=5)
            num_delayed_payment = st.number_input("Num of Delayed Payment", min_value=0, max_value=50, step=1, value=3)
            changed_credit_limit = st.number_input("Changed Credit Limit", min_value=0.0, step=0.1, value=5.0)
            num_credit_inquiries = st.number_input("Num Credit Inquiries", min_value=0, max_value=30, step=1, value=3)
            credit_mix = st.selectbox("Credit Mix", ["Standard", "Good", "Bad"])
            outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, step=10.0, value=500.0)
            credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, step=0.1, value=25.0)
            payment_of_min_amount = st.selectbox("Payment of Min Amount", ["Yes", "No", "NM"])
            total_emi_per_month = st.number_input("Total EMI per Month", min_value=0.0, step=10.0, value=50.0)
            amount_invested_monthly  = st.number_input("Amount Invested Monthly", min_value=0.0, step=10.0, value=100.0)
            payment_behaviour = st.selectbox("Payment Behaviour", [
                "Low_spent_Small_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Large_value_payments",
                "High_spent_Small_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Large_value_payments",
            ])
            monthly_balance = st.number_input("Monthly Balance", min_value=0.0, step=10.0, value=300.0)

        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        data = {
            "Age" : age,
            "Occupation" : occupation,
            "Annual_Income" : annual_income,
            "Monthly_Inhand_Salary" : monthly_inhand_salary,
            "Num_Bank_Accounts" : num_bank_accounts,
            "Num_Credit_Card" : num_credit_card,
            "Interest_Rate" : interest_rate,
            "Num_of_Loan" : num_of_loan,
            "Delay_from_due_date" : delay_from_due_date,
            "Num_of_Delayed_Payment" : num_delayed_payment,
            "Changed_Credit_Limit" : changed_credit_limit,
            "Num_Credit_Inquiries" : num_credit_inquiries,
            "Credit_Mix" : credit_mix,
            "Outstanding_Debt" : outstanding_debt,
            "Credit_Utilization_Ratio": credit_utilization_ratio,
            "Payment_of_Min_Amount" : payment_of_min_amount,
            "Total_EMI_per_month": total_emi_per_month,
            "Amount_invested_monthly": amount_invested_monthly,
            "Payment_Behaviour": payment_behaviour,
            "Monthly_Balance" : monthly_balance,
        }

        df = pd.DataFrame([data])
        pred_enc = model.predict(df)[0]
        prediction = le_target.inverse_transform([pred_enc])[0]

        st.divider()
        st.subheader("Hasil Prediksi")

        if prediction == "Good":
            st.success("Credit Score: **Good**")
            st.write("Nasabah memiliki riwayat kredit yang baik dan dianggap layak untuk mendapatkan kredit dengan bunga rendah.")
        elif prediction == "Standard":
            st.warning("Credit Score: **Standard**")
            st.write("Nasabah memiliki riwayat kredit yang cukup. Perlu evaluasi lebih lanjut sebelum pemberian kredit.")
        else:
            st.error("Credit Score: **Poor**")
            st.write("Nasabah memiliki riwayat kredit yang buruk. Berisiko tinggi untuk pemberian kredit.")


if __name__ == "__main__":
    main()