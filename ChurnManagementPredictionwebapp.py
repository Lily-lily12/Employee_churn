import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/user/OneDrive/Desktop/Employee_churn/train_model.sav', 'rb'))

# creating a function for Prediction
def churn_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person stays in the company'
    else:
        return 'The person left the company'

def main():
    # giving a title
    st.title('Churn Management Web App')

    # getting the input data from the user
    satisfaction_level = st.text_input('What is the satisfaction level on a scale of 0 to 1?', '0.5')
    last_evaluation = st.text_input('What is the last evaluation value on a scale of 0 to 1?', '0.5')
    number_project = st.text_input('What is the number of project undertaken?', '3')
    average_montly_hours = st.text_input('Average monthly hours', '160')
    time_spend_company = st.text_input('What is their time spent in the company?', '3')
    Work_accident = st.text_input('State the number of work accidents', '0')
    promotion_last_5years = st.text_input('Has the employee received a promotion in the last 5 years?', '0')

    # Using checkboxes for boolean values
    low = st.checkbox('Is the salary low?', value=False)
    medium = st.checkbox('Is the salary medium?', value=False)

    # Converting checkbox values to integer (1 for True, 0 for False)
    low = 1 if low else 0
    medium = 1 if medium else 0

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('Left or Stayed'):
        try:
            input_data = [
                float(satisfaction_level),
                float(last_evaluation),
                int(number_project),
                float(average_montly_hours),
                int(time_spend_company),
                int(Work_accident),
                int(promotion_last_5years),
                low,
                medium
            ]
            diagnosis = churn_prediction(input_data)
        except ValueError as e:
            diagnosis = f"Invalid input: {e}"

    st.success(diagnosis)

if __name__ == '__main__':
    main()
