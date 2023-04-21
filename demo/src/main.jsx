import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

const isProduction = process.env.NODE_ENV === "production";

ReactDOM.createRoot(document.getElementById("root")).render(
    <React.StrictMode>
        <App isProduction={true} />
    </React.StrictMode>
);
