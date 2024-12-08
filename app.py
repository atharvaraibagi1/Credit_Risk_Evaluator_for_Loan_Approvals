import pandas as pd
import numpy as np
import streamlit as st
import pickle

with open('loan_approval_xgboost.pkl', 'rb') as file:
    model = pickle.load(file)

final_numerical_woe = pd.read_excel("Numerical_WOE.xlsx")
final_categorical_woe = pd.read_excel("Categorical_WOE.xlsx")


def predict():
    income = st.numberinput("Enter Income:")   
    loan_amount = st.numberinput("Enter Loan Amount:")
    loan_int_rate = st.numberinput("Enter Loan Interest Rate:")
    person_home_ownership = st.selectbox("Enter Home Ownership Type:", ('RENT', 'MORTGAGE', 'OWN', 'OTHER'))
    credit_hist_length = st.numberinput("Enter CIBIL History Length:")
    loan_intent = st.selectbox("Enter Intent for loan:", ("EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"))

    person_financial_stability = income/ loan_amount
    debt_repayment_capacity = income - (loan_amount * loan_int_rate/100)
    credit_utilization_ratio = loan_amount / (income * credit_hist_length)
    person_interest_rate_adj_risk = loan_int_rate/100 * loan_amount
    person_income_credit_history= income / credit_hist_length

    # WOE for person_home_ownership
    home_df = final_categorical_woe[final_categorical_woe['feature']=='person_home_ownership']
    person_home_ownership_woe = home_df[home_df['value'] == person_home_ownership]['WOE'].values[0]

    # WOE for loan_intent
    loan_intent_df = final_categorical_woe[final_categorical_woe['feature']=='loan_intent']
    loan_intent_woe = loan_intent_df[loan_intent_df['value'] == loan_intent]['WOE'].values[0]

    # WOE for person_financial_stability
    pfs_df = final_numerical_woe[final_numerical_woe['feature']=='person_financial_stability'][['feature','lower_limit','upper_limit','WoE']].reset_index()
    for idx, i in enumerate(pfs_df['feature']):
        if person_financial_stability > pfs_df['lower_limit'][idx] and person_financial_stability <= pfs_df['upper_limit'][idx]:
            person_financial_stability_woe = pfs_df['WoE'][idx]
            
    # WOE for loan_int_rate
    lir_df = final_numerical_woe[final_numerical_woe['feature']=='loan_int_rate'][['feature','lower_limit','upper_limit','WoE']].reset_index()
    for idx, i in enumerate(lir_df['feature']):
        if loan_int_rate > lir_df['lower_limit'][idx] and loan_int_rate <= lir_df['upper_limit'][idx]:
            loan_int_rate_woe = lir_df['WoE'][idx]
            
    # WOE for debt_repayment_capacity
    drc_df = final_numerical_woe[final_numerical_woe['feature']=='debt_repayment_capacity'][['feature','lower_limit','upper_limit','WoE']].reset_index()
    for idx, i in enumerate(drc_df['feature']):
        if debt_repayment_capacity > drc_df['lower_limit'][idx] and debt_repayment_capacity <= drc_df['upper_limit'][idx]:
            debt_repayment_capacity_woe = drc_df['WoE'][idx]
            
    # WOE for credit_utilization_ratio
    cur_df = final_numerical_woe[final_numerical_woe['feature']=='credit_utilization_ratio'][['feature','lower_limit','upper_limit','WoE']].reset_index()
    for idx, i in enumerate(cur_df['feature']):
        if credit_utilization_ratio > cur_df['lower_limit'][idx] and credit_utilization_ratio <= cur_df['upper_limit'][idx]:
            credit_utilization_ratio_woe = cur_df['WoE'][idx]
            
    # WOE for person_interest_rate_adj_risk
    irar_df = final_numerical_woe[final_numerical_woe['feature']=='person_interest_rate_adj_risk'][['feature','lower_limit','upper_limit','WoE']].reset_index()
    for idx, i in enumerate(irar_df['feature']):
        if person_interest_rate_adj_risk > irar_df['lower_limit'][idx] and person_interest_rate_adj_risk <= irar_df['upper_limit'][idx]:
            person_interest_rate_adj_risk_woe = irar_df['WoE'][idx]
            
    # WOE for person_income_credit_history
    ich_df = final_numerical_woe[final_numerical_woe['feature']=='person_income_credit_history'][['feature','lower_limit','upper_limit','WoE']].reset_index()
    for idx, i in enumerate(ich_df['feature']):
        if person_income_credit_history > ich_df['lower_limit'][idx] and person_income_credit_history <= ich_df['upper_limit'][idx]:
            person_income_credit_history_woe = ich_df['WoE'][idx]    


    features = np.array([[
        person_financial_stability_woe,
        loan_int_rate_woe,
        debt_repayment_capacity_woe,
        person_home_ownership_woe,
        credit_utilization_ratio_woe,
        person_interest_rate_adj_risk_woe,
        person_income_credit_history_woe,
        loan_intent_woe
    ]])
    

    if st.button("Predict"):
        try:
            pred = model.predict(features)
            
            if pred == 0:
                pred_label = "Loan Rejected"
            else:
                pred_label = "Loan Approved"

            st.write(pred_label)
        except:
            st.write("An Error Occured")
def main():
    st.title("Credit Risk Evaluator for Loan Approvals")
    predict()

if __name__ == '__main__':
    main()