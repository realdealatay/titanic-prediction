import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import customtkinter as ct
from tkinter import messagebox

# Load data
df = pd.read_csv('titanic.csv')
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Split the data into training and testing sets
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Define the UI
class PassengerForm(ct.CTk):
    def __init__(self):
        super().__init__()

        self.title("Passenger Information Form")
        self.geometry("530x360")  # Increased height for additional space
        self.configure(background='#F4F6F9')  # Light background color

        ct.set_appearance_mode("Light")  # Set to Light mode for a modern look
        ct.set_default_color_theme("blue")  # Use a blue color theme

        self.create_form()

    def create_form(self):
        form_frame = ct.CTkFrame(self, corner_radius=15, fg_color="#FFFFFF")  # White background with rounded corners
        form_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Classe du billet
        ct.CTkLabel(form_frame, text="Ticket Class:", font=("Segoe UI", 14, "bold"), anchor="w").grid(row=0, column=0, sticky="w", padx=15, pady=(10, 5))
        self.classe_var = ct.StringVar(value="1")
        ct.CTkRadioButton(form_frame, text="1", variable=self.classe_var, value="1", font=("Segoe UI", 12)).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ct.CTkRadioButton(form_frame, text="2", variable=self.classe_var, value="2", font=("Segoe UI", 12)).grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ct.CTkRadioButton(form_frame, text="3", variable=self.classe_var, value="3", font=("Segoe UI", 12)).grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # Sexe du passager
        ct.CTkLabel(form_frame, text="Passenger's Gender:", font=("Segoe UI", 14, "bold"), anchor="w").grid(row=1, column=0, sticky="w", padx=15, pady=(10, 5))
        self.sexe_var = ct.StringVar(value="male")
        ct.CTkRadioButton(form_frame, text="Male", variable=self.sexe_var, value="male", font=("Segoe UI", 12)).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ct.CTkRadioButton(form_frame, text="Female", variable=self.sexe_var, value="female", font=("Segoe UI", 12)).grid(row=1, column=2, padx=5, pady=5, sticky="w")

        # Âge du passager
        ct.CTkLabel(form_frame, text="Passenger's age:", font=("Segoe UI", 14, "bold"), anchor="w").grid(row=2, column=0, sticky="w", padx=15, pady=(10, 5))
        self.age_entry = ct.CTkEntry(form_frame, width=250, height=35, corner_radius=10, font=("Segoe UI", 12))
        self.age_entry.grid(row=2, column=1, columnspan=3, padx=5, pady=5, sticky="w")

        # Nombre de frères et sœurs / conjoints
        ct.CTkLabel(form_frame, text="Brothers and Sisters:", font=("Segoe UI", 14, "bold"), anchor="w").grid(row=3, column=0, sticky="w", padx=15, pady=(10, 5))
        self.sibsp_entry = ct.CTkEntry(form_frame, width=250, height=35, corner_radius=10, font=("Segoe UI", 12))
        self.sibsp_entry.grid(row=3, column=1, columnspan=3, padx=5, pady=5, sticky="w")

        # Nombre de parents / enfants
        ct.CTkLabel(form_frame, text="Parents / children:", font=("Segoe UI", 14, "bold"), anchor="w").grid(row=4, column=0, sticky="w", padx=15, pady=(10, 5))
        self.parch_entry = ct.CTkEntry(form_frame, width=250, height=35, corner_radius=10, font=("Segoe UI", 12))
        self.parch_entry.grid(row=4, column=1, columnspan=3, padx=5, pady=5, sticky="w")

        # Port d'embarquement
        ct.CTkLabel(form_frame, text="Port of Embarkation:", font=("Segoe UI", 14, "bold"), anchor="w").grid(row=5, column=0, sticky="w", padx=15, pady=(10, 5))
        self.embarked_var = ct.StringVar(value="S")
        ct.CTkRadioButton(form_frame, text="S", variable=self.embarked_var, value="S", font=("Segoe UI", 12)).grid(row=5, column=1, padx=5, pady=5, sticky="w")
        ct.CTkRadioButton(form_frame, text="C", variable=self.embarked_var, value="C", font=("Segoe UI", 12)).grid(row=5, column=2, padx=5, pady=5, sticky="w")
        ct.CTkRadioButton(form_frame, text="Q", variable=self.embarked_var, value="Q", font=("Segoe UI", 12)).grid(row=5, column=3, padx=5, pady=5, sticky="w")

        # Submit button
        submit_button = ct.CTkButton(form_frame, text="Submit", command=self.predict_survival, font=("Segoe UI", 14, "bold"), corner_radius=10)
        submit_button.grid(row=6, column=0, columnspan=4, pady=20)


    def predict_survival(self):
        # Retrieve values from form
        pclass = int(self.classe_var.get())
        sex = self.sexe_var.get()
        age = float(self.age_entry.get() or 0)  # Default to 0 if empty
        sibsp = int(self.sibsp_entry.get() or 0)  # Default to 0 if empty
        parch = int(self.parch_entry.get() or 0)  # Default to 0 if empty
        embarked = self.embarked_var.get()

        # Map sex to numerical value
        sex_map = {'male': 0, 'female': 1}
        sex_value = sex_map.get(sex, 0)  # Default to 0 if sex is not in the map

        # Create DataFrame for the input data
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex_value],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [0],  # Assuming fare is not provided
            'Embarked': [embarked]
        })

        # One-hot encode the Embarked column
        input_data = pd.get_dummies(input_data, columns=['Embarked'], drop_first=True)

        # Ensure the input data has the same columns as the training data
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        # Make prediction
        prediction = knn.predict(input_data)[0]
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Show result using tkinter messagebox
        result = "will survive" if prediction == 1 else "won't survive"
        messagebox.showinfo(title="Prediction results:", message=f"The passenger {result} with a precision of {accuracy * 100:.2f}%")

# Create the application instance and run it
if __name__ == "__main__":
    app = PassengerForm()
    app.mainloop()
