# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from rich.console import Console
from rich.progress import Progress

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Create a console object
console = Console()

# Function to preprocess data
def preprocess(dataframe, is_train=True):
    console.log("Preprocessing data...")
    cat_cols = ['protocol_type', 'service', 'flag']
    num_cols = [col for col in dataframe.columns if col not in cat_cols + ['outcome', 'level']]
    
    # Handle categorical columns
    if is_train:
        console.log("Encoding categorical columns...")
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = pd.DataFrame(encoder.fit_transform(dataframe[cat_cols]))
        encoded.columns = encoder.get_feature_names_out(cat_cols)
        dataframe = dataframe.drop(cat_cols, axis=1).reset_index(drop=True)
        dataframe = pd.concat([dataframe, encoded], axis=1)
        joblib.dump(encoder, "encoder.pkl")
    else:
        console.log("Loading encoder for categorical columns...")
        encoder = joblib.load("encoder.pkl")
        encoded = pd.DataFrame(encoder.transform(dataframe[cat_cols]))
        encoded.columns = encoder.get_feature_names_out(cat_cols)
        dataframe = dataframe.drop(cat_cols, axis=1).reset_index(drop=True)
        dataframe = pd.concat([dataframe, encoded], axis=1)
    
    # Scale numerical columns
    if is_train:
        console.log("Scaling numerical columns...")
        scaler = RobustScaler()
        dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
        joblib.dump(scaler, "scaler.pkl")
    else:
        console.log("Loading scaler for numerical columns...")
        scaler = joblib.load("scaler.pkl")
        dataframe[num_cols] = scaler.transform(dataframe[num_cols])
    
    return dataframe

# Train and save the model
def train_and_save_model(data_path):
    console.log("Loading data from file...")
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'outcome', 'level'
    ]
    
    data = pd.read_csv(data_path, names=columns)
    console.log("Converting outcome column to binary labels...")
    data['outcome'] = data['outcome'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Preprocess the data
    data = preprocess(data)
    
    # Separate features and target
    console.log("Splitting data into features and target...")
    X = data.drop(['outcome', 'level'], axis=1).values
    y = data['outcome'].values
    
    # Split the data
    console.log("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with Progress() as progress:
        train_task = progress.add_task("[green]Training models...", total=100)
        
        # Train a Random Forest model 
        console.log("Training Random Forest model...")
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, "rf_model.pkl")
        progress.update(train_task, advance=50)
        
        # Train a Decision Tree model
        console.log("Training Decision Tree model...")
        dt_model = DecisionTreeClassifier(max_depth=3)
        dt_model.fit(X_train, y_train)
        joblib.dump(dt_model, "dt_model.pkl")
        progress.update(train_task, advance=50)
    
    # Evaluate models
    console.log("Evaluating models...")
    rf_predictions = rf_model.predict(X_test)
    dt_predictions = dt_model.predict(X_test)
    console.log("[bold yellow]Random Forest Classification Report:[/bold yellow]\n" + classification_report(y_test, rf_predictions))
    console.log("[bold yellow]Decision Tree Classification Report:[/bold yellow]\n" + classification_report(y_test, dt_predictions))

    console.log("[green]Models saved successfully![/green]")

# Predict function
def predict(input_data):
    console.log("Loading models and preprocessors for prediction...")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    rf_model = joblib.load("rf_model.pkl")
    dt_model = joblib.load("dt_model.pkl")
    
    console.log("Preprocessing input data...")
    input_data = preprocess(input_data, is_train=False)
    
    console.log("Making predictions...")
    rf_prediction = rf_model.predict(input_data)
    dt_prediction = dt_model.predict(input_data)
    
    return {"Random Forest": rf_prediction, "Decision Tree": dt_prediction}

# Main section with user interaction
if __name__ == "__main__":
    console.log("[bold cyan]Starting Manual Intrusion Detection System (IDS)...[/bold cyan]")
    train_and_save_model("nsl-kdd/KDDTrain+.txt")
    # Predefined test data corresponding to user choices
    predefined_data = [
        {'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF', 
        'src_bytes': 232, 'dst_bytes': 8153, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': 0, 'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0, 'root_shell': 0, 
        'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 
        'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 
        'count': 5, 'srv_count': 5, 'serror_rate': 0.2, 'srv_serror_rate': 0.2, 'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
        'dst_host_count': 30, 'dst_host_srv_count': 255, 'dst_host_same_srv_rate': 1.0, 
        'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 0.03, 
        'dst_host_srv_diff_host_rate': 0.04, 'dst_host_serror_rate': 0.03, 
        'dst_host_srv_serror_rate': 0.01, 'dst_host_rerror_rate': 0.0, 
        'dst_host_srv_rerror_rate': 0.01},
        {'duration': 0, 'protocol_type': 'tcp', 'service': 'private', 'flag': 'S0', 
        'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0, 
        'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 
        'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 
        'count': 117, 'srv_count': 16, 'serror_rate': 1.00, 'srv_serror_rate': 1.00, 
        'rerror_rate': 0.00, 'srv_rerror_rate': 0.00, 'same_srv_rate': 0.14, 
        'diff_srv_rate': 0.06, 'srv_diff_host_rate': 0.00, 'dst_host_count': 255, 
        'dst_host_srv_count': 15, 'dst_host_same_srv_rate': 0.06, 
        'dst_host_diff_srv_rate': 0.07, 'dst_host_same_src_port_rate': 0.00, 
        'dst_host_srv_diff_host_rate': 0.00, 'dst_host_serror_rate': 1.00, 
        'dst_host_srv_serror_rate': 1.00, 'dst_host_rerror_rate': 0.00, 
        'dst_host_srv_rerror_rate': 0.00},
        {'duration': 0, 'protocol_type': 'tcp', 'service': 'remote_job', 'flag': 'S0', 
        'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0, 
        'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 
        'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 
        'count': 270, 'srv_count': 23, 'serror_rate': 1.00, 'srv_serror_rate': 1.00, 
        'rerror_rate': 0.00, 'srv_rerror_rate': 0.00, 'same_srv_rate': 0.09, 
        'diff_srv_rate': 0.05, 'srv_diff_host_rate': 0.00, 'dst_host_count': 255, 
        'dst_host_srv_count': 23, 'dst_host_same_srv_rate': 0.09, 
        'dst_host_diff_srv_rate': 0.05, 'dst_host_same_src_port_rate': 0.00, 
        'dst_host_srv_diff_host_rate': 0.00, 'dst_host_serror_rate': 1.00, 
        'dst_host_srv_serror_rate': 1.00, 'dst_host_rerror_rate': 0.00, 
        'dst_host_srv_rerror_rate': 0.00},
        {'duration': 0, 'protocol_type': 'tcp', 'service': 'private', 'flag': 'S0', 
        'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0, 
        'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 
        'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 
        'count': 133, 'srv_count': 8, 'serror_rate': 1.00, 'srv_serror_rate': 1.00, 
        'rerror_rate': 0.00, 'srv_rerror_rate': 0.00, 'same_srv_rate': 0.06, 
        'diff_srv_rate': 0.06, 'srv_diff_host_rate': 0.00, 'dst_host_count': 255, 
        'dst_host_srv_count': 13, 'dst_host_same_srv_rate': 0.05, 
        'dst_host_diff_srv_rate': 0.06, 'dst_host_same_src_port_rate': 0.00, 
        'dst_host_srv_diff_host_rate': 0.00, 'dst_host_serror_rate': 1.00, 
        'dst_host_srv_serror_rate': 1.00, 'dst_host_rerror_rate': 0.00, 
        'dst_host_srv_rerror_rate': 0.00},
        {'duration': 0, 'protocol_type': 'tcp', 'service': 'private', 'flag': 'REJ', 
        'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0, 
        'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 
        'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 
        'count': 205, 'srv_count': 12, 'serror_rate': 0.00, 'srv_serror_rate': 0.00, 
        'rerror_rate': 1.00, 'srv_rerror_rate': 1.00, 'same_srv_rate': 0.06, 
        'diff_srv_rate': 0.06, 'srv_diff_host_rate': 0.00, 'dst_host_count': 255, 
        'dst_host_srv_count': 12, 'dst_host_same_srv_rate': 0.05, 
        'dst_host_diff_srv_rate': 0.07, 'dst_host_same_src_port_rate': 0.00, 
        'dst_host_srv_diff_host_rate': 0.00, 'dst_host_serror_rate': 0.00, 
        'dst_host_srv_serror_rate': 0.00, 'dst_host_rerror_rate': 1.00, 
        'dst_host_srv_rerror_rate': 1.00},
        {'duration': 0, 'protocol_type': 'tcp', 'service': 'private', 'flag': 'S0', 
        'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': 0, 'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0, 'root_shell': 0, 
        'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 
        'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 
        'count': 199, 'srv_count': 3, 'serror_rate': 1.00, 'srv_serror_rate': 1.00, 
        'rerror_rate': 0.00, 'srv_rerror_rate': 0.00, 'same_srv_rate': 0.02, 
        'diff_srv_rate': 0.06, 'srv_diff_host_rate': 0.00, 'dst_host_count': 255, 
        'dst_host_srv_count': 13, 'dst_host_same_srv_rate': 0.05, 
        'dst_host_diff_srv_rate': 0.07, 'dst_host_same_src_port_rate': 0.00, 
        'dst_host_srv_diff_host_rate': 0.00, 'dst_host_serror_rate': 1.00, 
        'dst_host_srv_serror_rate': 1.00, 'dst_host_rerror_rate': 0.00, 
        'dst_host_srv_rerror_rate': 0.00},
        {'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF', 
        'src_bytes': 287, 'dst_bytes': 2251, 'land': 0, 'wrong_fragment': 0, 'urgent': 0,
        'hot': 0, 'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0, 'root_shell': 0, 
        'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 
        'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 
        'count': 3, 'srv_count': 7, 'serror_rate': 0.00, 'srv_serror_rate': 0.00, 
        'rerror_rate': 0.00, 'srv_rerror_rate': 0.00, 'same_srv_rate': 1.00, 
        'diff_srv_rate': 0.00, 'srv_diff_host_rate': 0.43, 'dst_host_count': 8, 
        'dst_host_srv_count': 219, 'dst_host_same_srv_rate': 1.00, 
        'dst_host_diff_srv_rate': 0.00, 'dst_host_same_src_port_rate': 0.12, 
        'dst_host_srv_diff_host_rate': 0.03, 'dst_host_serror_rate': 0.00, 
        'dst_host_srv_serror_rate': 0.00, 'dst_host_rerror_rate': 0.00, 
        'dst_host_srv_rerror_rate': 0.00}
    ]

    # Load the trained models and preprocessors
    console.log("[bold cyan]Loading models and preprocessors...[/bold cyan]")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    rf_model = joblib.load("rf_model.pkl")
    dt_model = joblib.load("dt_model.pkl")

    while True:
        # Display the menu
        console.print("\n[bold yellow]Manual IDS Menu[/bold yellow]")
        console.print("Enter a number (1-10) to simulate data:")
        console.print("1 - 10: Predefined scenarios")
        console.print("0: Exit")

        # Get user input
        user_input = input("Your choice: ")

        try:
            choice = int(user_input)
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
            continue

        # Exit condition
        if choice == 0:
            console.log("[bold cyan]Exiting Manual IDS. Goodbye![/bold cyan]")
            break

        # Check if choice is valid
        if 1 <= choice <= len(predefined_data):
            console.log(f"[bold cyan]Processing choice {choice}...[/bold cyan]")
            
            # Fetch predefined data based on choice
            test_data = pd.DataFrame([predefined_data[choice - 1]])
            
            # Preprocess input data
            console.log("Preprocessing input data...")
            test_data = preprocess(test_data, is_train=False)

            # Make predictions
            console.log("Making predictions...")
            rf_prediction = rf_model.predict(test_data)[0]
            dt_prediction = dt_model.predict(test_data)[0]

            # Display results
            rf_label = "Normal" if rf_prediction == 0 else "Attack"
            dt_label = "Normal" if dt_prediction == 0 else "Attack"
            console.print(f"[green]Random Forest Prediction:[/green] {rf_label}")
            console.print(f"[green]Decision Tree Prediction:[/green] {dt_label}")
        else:
            console.print("[red]Invalid choice. Please select a number between 1 and 10.[/red]")