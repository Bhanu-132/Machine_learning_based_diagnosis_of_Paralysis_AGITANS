
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ttkthemes import ThemedTk

class ParkinsonsPredictorApp:
    def __init__(self, root):
        self.root = ThemedTk(theme="arc")  # Set the theme to "arc" (you can choose a different theme)
        self.root.title("Parkinson's Disease Predictor")

        # Create XGBoost classifier
        self.model = xgb.XGBClassifier()

        # Styling
        self.root.configure(bg='#E0E0E0')
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 12), padding=5, background='#4CAF50', foreground='black')
        style.configure('TLabel', font=('Helvetica', 12), padding=5, background='#E0E0E0')
        style.configure('TEntry', font=('Helvetica', 12), padding=5)
        style.map('TButton', background=[('active', '#45a049')])

        # Initialize X and y_test as class attributes
        self.X = None
        self.y_test = None

        # UI Components
        self.upload_button = ttk.Button(root, text="Upload Dataset", command=self.upload_dataset, style='TButton')
        self.upload_button.pack(pady=20)

        self.predict_button = ttk.Button(root, text="Predict Real-Time", command=self.predict_real_time, style='TButton')
        self.predict_button.pack(pady=20)

        self.visualization_button = ttk.Button(root, text="Show Visualization", command=self.show_visualization, style='TButton')
        self.visualization_button.pack(pady=20)

        self.accuracy_label = ttk.Label(root, text="", style='TLabel')
        self.accuracy_label.pack(pady=20)

        # Create a Figure instance for visualization
        self.fig, self.ax = plt.subplots(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def upload_dataset(self):
        file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

        if file_path:
            try:
                # Load the dataset
                data = pd.read_csv(file_path)

                # Set X as a class attribute
                self.X = data.drop('Parkinsons', axis=1)

                # Split data into features and labels
                X = self.X
                y = data['Parkinsons']

                # Split data into training and testing sets
                X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                self.model.fit(X_train, y_train)

                # Update model accuracy label

                accuracy = accuracy_score(self.y_test, self.model.predict(self.X_test))
                self.accuracy_label.config(text=f"Model Accuracy: {accuracy:.2%}")

                messagebox.showinfo("Success", "Dataset uploaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error uploading dataset: {str(e)}")

    def predict_real_time(self):
        try:
            # Simulate real-time data collection (you would replace this with actual data)
            random_data_point = pd.DataFrame({'MDVP:Fo(Hz)': [120.0],
                                              'MDVP:Fhi(Hz)': [150.0],
                                              'MDVP:Flo(Hz)': [80.0],
                                              'MDVP:Jitter(%)': [0.008],
                                              'MDVP:Shimmer': [0.05],
                                              'MDVP:Shimmer(dB)': [0.5],
                                              'Shimmer:APQ3': [0.03],
                                              'Shimmer:APQ5': [0.04],
                                              'MDVP:APQ': [0.04],
                                              'Shimmer:DDA': [0.07],
                                              'NHR': [0.02],
                                              'HNR': [20.0],
                                              'RPDE': [0.45],
                                              'DFA': [0.82],
                                              'spread1': [-4.8],
                                              'spread2': [2.3],
                                              'D2': [0.3],
                                              'PPE': [0.3]})
            prediction = self.model.predict(random_data_point)

            result_text = "Real-time Prediction: "
            if prediction[0] == 1:
                result_text += "The model predicts that Parkinson's disease is likely present based on the input data."
            else:
                result_text += "The model predicts that Parkinson's disease is unlikely based on the input data."
            
            messagebox.showinfo("Prediction", result_text)
        except Exception as e:
            messagebox.showerror("Error", f"Error predicting disease in real-time: {str(e)}")

    def show_visualization(self):
        try:
            # Check if X is defined
            if self.X is not None:
                # Create a scatter plot
                colors = ['red' if label == 1 else 'green' for label in self.model.predict(self.X)]
                self.ax.scatter(self.X.iloc[:, 0], self.X.iloc[:, 1], c=colors, alpha=0.5)
                self.ax.set_xlabel('Feature 1')
                self.ax.set_ylabel('Feature 2')
                self.ax.set_title('Scatter Plot for Parkinson\'s Disease Prediction')
                self.canvas.draw()
            else:
                messagebox.showwarning("Warning", "Please upload a dataset first.")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating visualization: {str(e)}")

if __name__ == "__main__":
    root = ThemedTk(theme="arc")  # Set the theme to "arc" (you can choose a different theme)
    app = ParkinsonsPredictorApp(root)
    root.mainloop()