# --- Step 5: Dashboard and Visualisation ---

from dash import Dash, dcc, html, dash_table
import plotly.express as px
import plotly.figure_factory as ff

from model import train_model


# Train model and collect results
model, data, accuracy, cm, importance = train_model()

# Key descriptive statistics
summary_stats = data.describe().round(2).reset_index()

summary_table = dash_table.DataTable(
    data=summary_stats.to_dict("records"),
    columns=[{"name": col, "id": col} for col in summary_stats.columns],
    page_size=10,
    style_table={"overflowX": "auto"},
    style_cell={"textAlign": "center", "padding": "8px"},
    style_header={"fontWeight": "bold"}
)

# Additional plots
price_fig = px.line(
    data,
    y=["Close", "MA5", "MA20"],
    title="Volvo B Stock Price and Moving Averages"
)

return_fig = px.histogram(
    data,
    x="Return",
    title="Distribution of Daily Returns"
)

# Correlation check
correlation = data[["Return", "MA5", "MA20", "Volatility", "Target"]].corr()

correlation_fig = px.imshow(
    correlation,
    text_auto=True,
    title="Correlation Matrix"
)

# Feature importance
importance_fig = px.bar(
    importance,
    x="Feature",
    y="Importance",
    title="Feature Importance"
)

# Model performance visual
cm_fig = ff.create_annotated_heatmap(
    z=cm,
    x=["Predicted Down", "Predicted Up"],
    y=["Actual Down", "Actual Up"]
)

cm_fig.update_layout(title="Confusion Matrix")

# Latest prediction
latest_X = data[["Return", "MA5", "MA20", "Volatility"]].iloc[-1:]
latest_prediction = model.predict(latest_X)[0]
prediction_text = "Up" if latest_prediction == 1 else "Down"

# Build Dash app
app = Dash(__name__)

app.layout = html.Div(
    style={"padding": "20px", "fontFamily": "Arial"},
    children=[
        html.H1("Volvo B Stock Prediction Dashboard"),

        html.H2("Project Question"),
        html.P("Can historical stock data predict if Volvo B will go up or down tomorrow?"),

        html.H2("Step 1: Exploratory Data Analysis"),
        dcc.Graph(figure=price_fig),
        dcc.Graph(figure=return_fig),
        html.H3("Key Descriptive Statistics"),
        summary_table,

        html.H2("Step 2: Feature Selection and Cleaning"),
        html.P(
            "The model uses daily return, MA5, MA20 and volatility as features. "
            "Missing values are removed, and the correlation matrix is used to inspect relationships between variables."
        ),
        dcc.Graph(figure=correlation_fig),

        html.H2("Step 3: Model Building"),
            html.P(
            "The predictive model used in this project is a Random Forest classifier. "
            "This is a supervised machine learning model used for classification problems."
        ),
            html.P(
            "The model is trained using four explanatory variables: daily return, MA5, MA20 and volatility. "
            "The target variable is whether the Volvo B stock price goes up or down the next trading day."
        ),
            html.P(
            "The dataset is split into training and testing data. The model learns from the training data "
            "and is later evaluated on unseen test data."
        ),

        html.H2("Step 4: Evaluation"),
        html.P(f"Accuracy: {accuracy:.2%}"),
        html.P(f"Latest prediction: {prediction_text}"),
        dcc.Graph(figure=importance_fig),
        dcc.Graph(figure=cm_fig),
    ]
)

if __name__ == "__main__":
    app.run(debug=True)