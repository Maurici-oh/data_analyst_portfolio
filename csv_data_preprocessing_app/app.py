import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
import numpy as np
import time
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.colors
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.feature_selection import SelectKBest, f_classif
import uuid
from copy import deepcopy  # For safely copying operation parameters


def main():
    # --- Initialize Session State ---
    if "initial_df" not in st.session_state:
        st.session_state["initial_df"] = None  # To store the loaded DataFrame
    if "uploaded_file_id" not in st.session_state:
        st.session_state["uploaded_file_id"] = (
            None  # To track the ID of the file currently loaded
        )
    if "operations_history" not in st.session_state:
        st.session_state["operations_history"] = (
            []
        )  # To store the history of applied operations
    if "current_df" not in st.session_state:
        st.session_state["current_df"] = None  # To store the current DataFrame
    if "initial_train_df" not in st.session_state:
        st.session_state["initial_train_df"] = None
    if "initial_validation_df" not in st.session_state:
        st.session_state["initial_validation_df"] = None
    if "initial_test_df" not in st.session_state:
        st.session_state["initial_test_df"] = None
    if "train_df" not in st.session_state:
        st.session_state["train_df"] = None
    if "validation_df" not in st.session_state:
        st.session_state["validation_df"] = None
    if "test_df" not in st.session_state:
        st.session_state["test_df"] = None
    if "message" not in st.session_state:
        st.session_state["message"] = []  # Store the operation message
    if "is_data_scaled" not in st.session_state:
        st.session_state.is_data_scaled = False
    if "is_data_encoded" not in st.session_state:
        st.session_state.is_data_encoded = False
    if "is_data_transformed" not in st.session_state:
        st.session_state.is_data_transformed = False
    if "is_data_split" not in st.session_state:
        st.session_state.is_data_split = False
    if "pca_done" not in st.session_state:
        st.session_state.pca_done = False
    if "verify_clicked" not in st.session_state:
        st.session_state.verify_clicked = False  # verify button session state (PCA)
    if "explained_variance" not in st.session_state:
        st.session_state["explained_variance"] = []  # Explained variance list after PCA
    if "debug" not in st.session_state:
        st.session_state["debug"] = None  # Debuggin session state

    # --- Sidebar Section for File Upload ---
    st.sidebar.title("🛠️ CSV Data Preprocessing App")
    st.sidebar.divider()
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file here", type=["csv"], key="file_uploader_widget"
    )

    # --- Data Loading and Cache Logic ---
    if uploaded_file is not None:
        # Check if the newly uploaded file is different from the one we've already processed
        if uploaded_file.file_id != st.session_state["uploaded_file_id"]:
            try:
                df = pd.read_csv(uploaded_file)

                # Store the loaded DataFrame and the file ID in session state
                st.session_state["initial_df"] = df
                st.session_state["uploaded_file_id"] = uploaded_file.file_id
                st.session_state["current_df"] = st.session_state["initial_df"].copy()
                # logging.info('DataFrame imported.')

            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                # Reset session state in case of error
                st.success(
                    "File removed. Clearing data.", icon=":material/delete_sweep:"
                )
                st.session_state["operations_history"] = []
                st.session_state["uploaded_file_id"] = None
                st.session_state["initial_df"] = None
                st.session_state["current_df"] = None
                st.session_state["initial_train_df"] = None
                st.session_state["initial_validation_df"] = None
                st.session_state["initial_test_df"] = None
                st.session_state["train_df"] = None
                st.session_state["validation_df"] = None
                st.session_state["test_df"] = None
                st.session_state["message"] = None
                st.session_state["explained_variance"] = []
                st.session_state.pca_done = False
                st.session_state.verify_clicked = False
                st.session_state.is_data_scaled = False
                st.session_state.is_data_encoded = False
                st.session_state.is_data_transformed = False
                st.session_state.is_data_split = False
        else:
            pass

    else:
        # If no file is uploaded (or it was removed), clear the session states
        if st.session_state["uploaded_file_id"] is not None:
            st.success("File removed. Clearing data.", icon=":material/delete_sweep:")
            st.session_state["operations_history"] = []
            st.session_state["uploaded_file_id"] = None
            st.session_state["initial_df"] = None
            st.session_state["current_df"] = None
            st.session_state["initial_train_df"] = None
            st.session_state["initial_validation_df"] = None
            st.session_state["initial_test_df"] = None
            st.session_state["train_df"] = None
            st.session_state["validation_df"] = None
            st.session_state["test_df"] = None
            st.session_state["message"] = None
            st.session_state["explained_variance"] = []
            st.session_state.pca_done = False
            st.session_state.verify_clicked = False
            st.session_state.is_data_scaled = False
            st.session_state.is_data_encoded = False
            st.session_state.is_data_transformed = False
            st.session_state.is_data_split = False

    # --- Preprocessing starts here ---
    if st.session_state["initial_df"] is not None:

        def look_for_scaler():
            history_list = st.session_state.get("operations_history", [])

            is_data_scaled = False  # Flag to track if we found the specific scaler
            scaler = None

            # Iterate through the items (dictionaries) in the list
            for operation in history_list:
                # Check if the current list item is a dictionary AND has 'op_type' and 'params' keys
                if (
                    isinstance(operation, dict)
                    and "op_type" in operation
                    and "params" in operation
                ):
                    op_type = operation["op_type"]
                    params = operation["params"]

                    if op_type == "scale_columns":
                        # Check if 'params' is a dictionary and has 'scaling_method'
                        if isinstance(params, dict) and "scaling_method" in params:
                            scaling_method = params["scaling_method"]
                            is_data_scaled = True
                            st.session_state.is_data_scaled = True

                            # Assign the scaling_method to scaler variable"
                            if scaling_method == "StandardScaler (Z-score)":
                                scaler = scaling_method
                            elif scaling_method == "MinMaxScaler (0-1 range)":
                                scaler = scaling_method
                            elif scaling_method == "Robust Scaler":
                                scaler = scaling_method
                                break

            # # Simple version of the look_for_scaler() function
            # history_list = st.session_state.get('operations_history', [])
            # for operation in history_list:
            #     if 'op_type' in operation:
            #         op_type = operation['op_type']
            #         if op_type == "scale_columns":
            #             st.session_state['is_data_scaled'] = True
            #             break

            return is_data_scaled, scaler

        def all_columns_numeric(df):
            return all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)

        def target_encode(df, categorical_col, target_col, smoothing=0.0):
            # Calculate the mean of the target for each category
            mean_target_per_category = df.groupby(categorical_col)[
                target_col
            ].transform("mean")
            # Calculate the global mean of the target
            global_mean_target = df[target_col].mean()
            if smoothing > 0:
                # Calculate counts for each category
                category_counts = df.groupby(categorical_col)[target_col].transform(
                    "count"
                )
                # Apply smoothing
                # Smoothed estimate = (mean_per_category * count + global_mean * smoothing) / (count + smoothing)
                encoded_series = (
                    (mean_target_per_category * category_counts)
                    + (global_mean_target * smoothing)
                ) / (category_counts + smoothing)
            else:
                # No smoothing, just use the mean per category
                encoded_series = mean_target_per_category

            return encoded_series

        def apply_operation(df, op_spec):
            """
            Applies a single data transformation operation to a given DataFrame.
            Returns the transformed DataFrame or the original if an error occurs.
            """
            op_type = op_spec["op_type"]
            params = op_spec["params"]
            is_split = st.session_state.get("is_data_split", False)
            df_input = df.copy()

            try:
                match op_type:
                    case "rename_column":
                        if is_split:
                            df_train_input = df_input
                            df_test_input = st.session_state["test_df"].copy()

                            if params["old_name"] in df_train_input.columns:

                                df_train_processed = df_train_input.rename(
                                    columns={params["old_name"]: params["new_name"]}
                                )
                                df_test_processed = df_test_input.rename(
                                    columns={params["old_name"]: params["new_name"]}
                                )

                                # Update session_state
                                st.session_state["current_df"] = (
                                    df_train_processed.copy()
                                )
                                st.session_state["test_df"] = df_test_processed.copy()

                                # Send message
                                message_text = f":material/check_circle: **{params['old_name']}** column renamed to **{params['new_name']}**"
                                message_type = "success"
                                st.session_state["message"] = (
                                    message_type,
                                    message_text,
                                )

                                if st.session_state["validation_df"] is not None:
                                    df_val_input = st.session_state[
                                        "validation_df"
                                    ].copy()
                                    df_val_processed = df_val_input.rename(
                                        columns={params["old_name"]: params["new_name"]}
                                    )

                                    st.session_state["validation_df"] = (
                                        df_val_processed.copy()
                                    )

                                    return (
                                        df_train_processed,
                                        df_val_processed,
                                        df_test_processed,
                                    )

                            return df_train_processed, df_test_processed

                        else:
                            if params["old_name"] in df_input.columns:
                                df_processed = df_input.rename(
                                    columns={params["old_name"]: params["new_name"]}
                                )

                                # Update session_state
                                st.session_state["current_df"] = df_processed.copy()

                                # Send message
                                message_text = f":material/check_circle: **{params['old_name']}** column renamed to **{params['new_name']}**"
                                message_type = "success"
                                st.session_state["message"] = (
                                    message_type,
                                    message_text,
                                )

                            return df_processed

                    case "remove_duplicates":
                        # This operation is not possible after split
                        df_processed = df_input

                        keep = params["keep_first"]
                        if keep == True:
                            rows_before = len(df_processed)
                            df_processed.drop_duplicates(
                                subset=None, keep="first", inplace=True
                            )
                            rows_after = len(df_processed)
                            message_type = "success"
                            message_text = f":material/check_circle: Dulicate rows removed successfully! Removed {rows_before - rows_after} rows."

                        elif keep == False:
                            rows_before = len(df_processed)
                            df_processed.drop_duplicates(
                                subset=None, keep=False, inplace=True
                            )
                            rows_after = len(df_processed)
                            message_type = "success"
                            message_text = f":material/check_circle: Dulicate rows removed successfully! Removed {rows_before - rows_after} rows."

                        # Update session_state
                        st.session_state["current_df"] = df_processed

                        # Send message
                        st.session_state["message"] = (message_type, message_text)

                        return df_processed

                    case "drop_columns":

                        if is_split:
                            df_train_input = df_input
                            df_test_input = st.session_state["test_df"].copy()

                            cols_to_drop = [
                                col
                                for col in params["columns"]
                                if col in df_train_input.columns
                            ]

                            df_train_processed = df_train_input
                            df_test_processed = df_test_input

                            if cols_to_drop:
                                df_train_processed = df_train_processed.drop(
                                    columns=cols_to_drop
                                )
                                df_test_processed = df_test_processed.drop(
                                    columns=cols_to_drop
                                )

                                # Update session_state
                                st.session_state["current_df"] = (
                                    df_train_processed.copy()
                                )
                                st.session_state["test_df"] = df_test_processed.copy()

                                # Send message
                                message_text = f":material/check_circle: {cols_to_drop} column(s) dropped successfully!"
                                message_type = "success"
                                st.session_state["message"] = (
                                    message_type,
                                    message_text,
                                )

                                if st.session_state["validation_df"] is not None:
                                    df_val_input = st.session_state[
                                        "validation_df"
                                    ].copy()
                                    df_val_processed = df_val_input
                                    df_val_processed = df_val_processed.drop(
                                        columns=cols_to_drop
                                    )
                                    st.session_state["validation_df"] = (
                                        df_val_processed.copy()
                                    )

                                    return (
                                        df_train_processed,
                                        df_val_processed,
                                        df_test_processed,
                                    )

                            return df_train_processed, df_test_processed

                        else:
                            cols_to_drop = [
                                col
                                for col in params["columns"]
                                if col in df_input.columns
                            ]
                            df_processed = df_input.copy()

                            if cols_to_drop:
                                df_processed = df_processed.drop(columns=cols_to_drop)
                                st.session_state["current_df"] = df_processed.copy()

                                message_text = f":material/check_circle: {cols_to_drop} column(s) dropped successfully!"
                                message_type = "success"
                                st.session_state["message"] = (
                                    message_type,
                                    message_text,
                                )

                            return df_processed

                    case "missing_values":
                        # This operation is not possible after split
                        df_processed = df_input

                        match params["fill_strategy"]:
                            case "Drop":
                                axis = params["axis"]
                                df_processed = df_input.dropna(axis=axis)
                                message_type = "success"
                                message_text = ":material/check_circle: Missing values dropped successfully!"
                            case "Fill with Mean":
                                columns_na = params["cols_with_na"]
                                for col in columns_na:
                                    df_processed[columns_na] = df_input[
                                        columns_na
                                    ].fillna(
                                        df_input[columns_na].mean(numeric_only=True)
                                    )
                                message_type = "success"
                                message_text = ":material/check_circle: Missing values filled with mean successfully!"
                            case "Fill with Median":
                                columns_na = params["cols_with_na"]
                                for col in columns_na:
                                    df_processed[columns_na] = df_input[
                                        columns_na
                                    ].fillna(
                                        df_input[columns_na].median(numeric_only=True)
                                    )
                                message_type = "success"
                                message_text = ":material/check_circle: Missing values filled with median successfully!"
                            case "Fill with Mode":
                                columns_na = params["cols_with_na"]
                                for col in columns_na:
                                    df_processed[columns_na] = df_input[
                                        columns_na
                                    ].fillna(df_input[columns_na].mode().iloc[0])
                                message_type = "success"
                                message_text = ":material/check_circle: Missing values filled with mode successfully!"
                            case "Fill with Custom Value":
                                df_processed = df_input.fillna(params["custom_value"])

                                message_type = "success"
                                message_text = f':material/check_circle: Missing values filled with "{custom_value}" successfully!'

                        # Update session_state
                        st.session_state["current_df"] = df_processed.copy()

                        # Send message
                        st.session_state["message"] = (message_type, message_text)

                        return df_processed

                    case "datatype_convertion":

                        if is_split:
                            # Get copies of the current train and test data from session state.
                            df_train_input = df_input
                            df_test_input = st.session_state["test_df"].copy()

                            df_train_processed = df_train_input.copy()
                            df_test_processed = df_test_input.copy()

                            selected_column = params["selected_column"]
                            target_datatype = params["selected_datatype"]

                            original_train_column_series = df_train_processed[
                                selected_column
                            ].copy()  # For comparison and potential revert
                            original_test_column_series = df_test_processed[
                                selected_column
                            ].copy()

                            if st.session_state["validation_df"] is not None:
                                df_val_input = st.session_state["validation_df"].copy()
                                df_val_processed = df_val_input.copy()
                                original_val_column_series = df_val_processed[
                                    selected_column
                                ].copy()

                            try:
                                if target_datatype == "int":
                                    # Attempt to convert to numeric first (errors='coerce' turns non-numeric to NaN)
                                    numeric_train_series = pd.to_numeric(
                                        original_train_column_series, errors="coerce"
                                    )
                                    numeric_test_series = pd.to_numeric(
                                        original_test_column_series, errors="coerce"
                                    )
                                    if st.session_state["validation_df"] is not None:
                                        numeric_val_series = pd.to_numeric(
                                            original_val_column_series, errors="coerce"
                                        )

                                    # Check if all values became NaN (and original wasn't all NaN)
                                    if (
                                        numeric_train_series.isnull().all()
                                        and not original_train_column_series.isnull().all()
                                    ):
                                        message_text = f":material/cancel: Could not convert any values in **{selected_column}** to numeric for integer conversion. All values resulted in NaN."
                                        message_type = "error"
                                        st.session_state["message"] = (
                                            message_type,
                                            message_text,
                                        )

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            return (
                                                df_train_input,
                                                df_val_input,
                                                df_test_input,
                                            )

                                        return df_train_input, df_test_input

                                    # Check for any NaNs. Standard 'int' (like np.int64) cannot have NaNs.
                                    if numeric_train_series.isnull().any():
                                        failed_train_conversions_count = (
                                            numeric_train_series.isnull()
                                            & original_train_column_series.notnull()
                                        ).sum()
                                        original_train_nans_count = (
                                            numeric_train_series.isnull()
                                            & original_train_column_series.isnull()
                                        ).sum()

                                        if failed_train_conversions_count > 0:
                                            message_text = f":material/warning: Column **{selected_column}** cannot be converted to **int**. {failed_train_conversions_count} value(s) were non-numeric and became NaN."
                                        else:  # Only original NaNs are present
                                            message_text = f":material/warning: Column **{selected_column}** cannot be converted to **int** as it contains {original_train_nans_count} original NaN/NaT value(s) which are not supported by standard integer types. "
                                        message_type = "warning"
                                        st.session_state["message"] = (
                                            message_type,
                                            message_text,
                                        )

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            return (
                                                df_train_input,
                                                df_val_input,
                                                df_test_input,
                                            )

                                        return (
                                            df_train_input,
                                            df_test_input,
                                        )  # Return original df as strict 'int' conversion failed

                                    else:
                                        # All values are numeric and non-null, safe to convert to standard int
                                        df_train_processed[selected_column] = (
                                            numeric_train_series.astype(np.int64)
                                        )
                                        df_test_processed[selected_column] = (
                                            numeric_test_series.astype(np.int64)
                                        )

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            df_val_processed[selected_column] = (
                                                numeric_val_series.astype(np.int64)
                                            )

                                        message_type = "success"
                                        message_text = f":material/check_circle: **{selected_column}** column converted to **{target_datatype}** successfully!"

                                elif target_datatype == "float":
                                    # Attempt to convert to numeric first (errors='coerce' turns non-numeric to NaN)
                                    numeric_train_series = pd.to_numeric(
                                        original_train_column_series, errors="coerce"
                                    )
                                    numeric_test_series = pd.to_numeric(
                                        original_test_column_series, errors="coerce"
                                    )
                                    if st.session_state["validation_df"] is not None:
                                        numeric_val_series = pd.to_numeric(
                                            original_val_column_series, errors="coerce"
                                        )

                                    # Check if all values became NaN (and original wasn't all NaN)
                                    if (
                                        numeric_train_series.isnull().all()
                                        and not original_train_column_series.isnull().all()
                                    ):
                                        message_text = f":material/cancel: Could not convert any values in **{selected_column}** to **float**. All values resulted in NaN."
                                        message_type = "error"
                                        st.session_state["message"] = (
                                            message_type,
                                            message_text,
                                        )

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            return (
                                                df_train_input,
                                                df_val_input,
                                                df_test_input,
                                            )

                                        return df_train_input, df_test_input

                                    df_train_processed[selected_column] = (
                                        numeric_train_series.astype(np.float64)
                                    )
                                    new_train_nans_from_values = (
                                        numeric_train_series.isnull()
                                        & original_train_column_series.notnull()
                                    ).sum()

                                    df_test_processed[selected_column] = (
                                        numeric_test_series.astype(np.float64)
                                    )
                                    new_test_nans_from_values = (
                                        numeric_test_series.isnull()
                                        & original_test_column_series.notnull()
                                    ).sum()

                                    if st.session_state["validation_df"] is not None:
                                        df_val_processed[selected_column] = (
                                            numeric_val_series.astype(np.float64)
                                        )
                                        new_val_nans_from_values = (
                                            numeric_val_series.isnull()
                                            & original_val_column_series.notnull()
                                        ).sum()

                                    if (
                                        new_train_nans_from_values > 0
                                        and new_test_nans_from_values > 0
                                    ):
                                        message_type = "warning"
                                        message_text = f":material/warning: **{selected_column}** column converted to **{target_datatype}**. However, {new_train_nans_from_values} and {new_test_nans_from_values} value(s) could not be converted to numeric and are now NaN."
                                    else:
                                        message_type = "success"
                                        message_text = f":material/check_circle: **{selected_column}** column converted to **{target_datatype}** successfully!"

                                elif target_datatype == "datetime":
                                    # Attempt to convert to datetime first (errors='coerce' turns non-numeric to NaN)
                                    datetime_train_series = pd.to_datetime(
                                        original_train_column_series, errors="coerce"
                                    )
                                    datetime_test_series = pd.to_datetime(
                                        original_test_column_series, errors="coerce"
                                    )

                                    if st.session_state["validation_df"] is not None:
                                        datetime_val_series = pd.to_datetime(
                                            original_val_column_series, errors="coerce"
                                        )

                                    if (
                                        datetime_train_series.isnull().all()
                                        and not original_train_column_series.isnull().all()
                                    ):
                                        message_text = (
                                            f":material/cancel: Could not convert any values in "
                                            f"**{selected_column}** to **datetime**. All values resulted in NaT."
                                        )
                                        message_type = "error"
                                        st.session_state["message"] = (
                                            message_type,
                                            message_text,
                                        )

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            return (
                                                df_train_input,
                                                df_val_input,
                                                df_test_input,
                                            )

                                        return df_train_input, df_test_input

                                    df_train_processed[selected_column] = (
                                        datetime_train_series
                                    )
                                    new_train_nats_from_values = (
                                        datetime_train_series.isnull()
                                        & original_train_column_series.notnull()
                                    ).sum()

                                    df_test_processed[selected_column] = (
                                        datetime_test_series
                                    )
                                    new_test_nats_from_values = (
                                        datetime_test_series.isnull()
                                        & original_test_column_series.notnull()
                                    ).sum()

                                    if st.session_state["validation_df"] is not None:
                                        df_val_processed[selected_column] = (
                                            datetime_val_series
                                        )
                                        new_val_nats_from_values = (
                                            datetime_val_series.isnull()
                                            & original_val_column_series.notnull()
                                        ).sum()

                                    if (
                                        new_train_nats_from_values > 0
                                        and new_test_nats_from_values > 0
                                    ):
                                        message_type = "warning"
                                        message_text = (
                                            f":material/warning: **{selected_column}** column converted to **{target_datatype}**. "
                                            f"However, {new_train_nats_from_values} and {new_test_nats_from_values} value(s) could not be converted to datetime and are now NaT."
                                        )
                                    else:
                                        message_type = "success"
                                        message_text = f":material/check_circle: **{selected_column}** column converted to **{target_datatype}** successfully!"

                                elif target_datatype == "category":
                                    num_train_unique = (
                                        original_train_column_series.nunique()
                                    )
                                    total_train_rows = len(original_train_column_series)

                                    # Heuristic thresholds for high cardinality
                                    high_cardinality_ratio = (
                                        num_train_unique / total_train_rows
                                    )
                                    high_cardinality = total_train_rows > 100 and (
                                        high_cardinality_ratio > 0.8
                                        or num_train_unique > 200
                                    )

                                    if high_cardinality:
                                        message_text = (
                                            f" :material/info: Column **{selected_column}** has high cardinality "
                                            f"({num_train_unique} unique values out of {total_train_rows}, {high_cardinality_ratio:.1%}). "
                                            f"Converting to 'category' may not offer significant memory benefits or could slow down performance."
                                        )
                                        message_type = "info"
                                        st.session_state["message"] = (
                                            message_type,
                                            message_text,
                                        )

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            return (
                                                df_train_input,
                                                df_val_input,
                                                df_test_input,
                                            )

                                        return df_train_input, df_test_input

                                    else:
                                        df_train_processed[selected_column] = (
                                            original_train_column_series.astype(
                                                "category"
                                            )
                                        )
                                        df_test_processed[selected_column] = (
                                            original_test_column_series.astype(
                                                "category"
                                            )
                                        )
                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            df_val_processed[selected_column] = (
                                                original_val_column_series.astype(
                                                    "category"
                                                )
                                            )

                                        message_type = "success"
                                        message_text = f":material/check_circle: **{selected_column}** column converted to **{target_datatype}** successfully!"

                                elif target_datatype == "str":
                                    df_train_processed[selected_column] = (
                                        original_train_column_series.astype(str)
                                    )
                                    df_test_processed[selected_column] = (
                                        original_test_column_series.astype(str)
                                    )

                                    if st.session_state["validation_df"] is not None:
                                        df_val_processed[selected_column] = (
                                            original_val_column_series.astype(str)
                                        )

                                    message_type = "success"
                                    message_text = f":material/check_circle: **{selected_column}** column converted to **{target_datatype}** successfully!"

                                elif target_datatype == "bool":
                                    # Check if already a boolean type (including pandas nullable BooleanDtype)
                                    if pd.api.types.is_bool_dtype(
                                        original_train_column_series.dtype
                                    ):
                                        # If it's already bool, no conversion needed, but ensure df_processed reflects this.
                                        df_train_processed[selected_column] = (
                                            original_train_column_series
                                        )
                                        df_test_processed[selected_column] = (
                                            original_test_column_series
                                        )

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            df_val_processed[selected_column] = (
                                                original_val_column_series
                                            )

                                        message_type = "success"
                                        message_text = f":material/check_circle: **{selected_column}** column is already **bool**."
                                    else:
                                        true_values = {
                                            "true",
                                            "1",
                                            "yes",
                                            "t",
                                            "y",
                                            "Yes",
                                            "Approved",
                                        }
                                        false_values = {
                                            "false",
                                            "0",
                                            "no",
                                            "f",
                                            "n",
                                            "No",
                                            "Rejected",
                                        }

                                        result_train_series = pd.Series(
                                            pd.NA,
                                            index=original_train_column_series.index,
                                            dtype=pd.BooleanDtype(),
                                        )
                                        unmappable_train_raw_values = []
                                        conversion_train_possible = True

                                        result_test_series = pd.Series(
                                            pd.NA,
                                            index=original_test_column_series.index,
                                            dtype=pd.BooleanDtype(),
                                        )
                                        unmappable_test_raw_values = []
                                        conversion_test_possible = True

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            result_val_series = pd.Series(
                                                pd.NA,
                                                index=original_val_column_series.index,
                                                dtype=pd.BooleanDtype(),
                                            )
                                            unmappable_val_raw_values = []
                                            conversion_val_possible = True

                                        for (
                                            idx,
                                            val,
                                        ) in original_train_column_series.items():
                                            if pd.isna(val):  # Preserve original NaNs
                                                result_train_series.loc[idx] = pd.NA
                                                continue

                                            val_train_str_lower = str(
                                                val
                                            ).lower()  # Convert value to lowercase string for comparison

                                            if val_train_str_lower in true_values:
                                                result_train_series.loc[idx] = True
                                            elif val_train_str_lower in false_values:
                                                result_train_series.loc[idx] = False
                                            else:
                                                # This value is not recognized as bool and is not NaN
                                                if (
                                                    len(unmappable_train_raw_values) < 5
                                                ):  # Collect a few examples of unmappable values
                                                    unmappable_train_raw_values.append(
                                                        val
                                                    )
                                                conversion_train_possible = False  # Flag that not all values are mappable

                                        for (
                                            idx,
                                            val,
                                        ) in original_test_column_series.items():
                                            if pd.isna(val):  # Preserve original NaNs
                                                result_test_series.loc[idx] = pd.NA
                                                continue

                                            val_test_str_lower = str(
                                                val
                                            ).lower()  # Convert value to lowercase string for comparison

                                            if val_test_str_lower in true_values:
                                                result_test_series.loc[idx] = True
                                            elif val_test_str_lower in false_values:
                                                result_test_series.loc[idx] = False
                                            else:
                                                # This value is not recognized as bool and is not NaN
                                                if (
                                                    len(unmappable_test_raw_values) < 5
                                                ):  # Collect a few examples of unmappable values
                                                    unmappable_test_raw_values.append(
                                                        val
                                                    )
                                                conversion_test_possible = False  # Flag that not all values are mappable

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            for (
                                                idx,
                                                val,
                                            ) in original_val_column_series.items():
                                                if pd.isna(
                                                    val
                                                ):  # Preserve original NaNs
                                                    result_val_series.loc[idx] = pd.NA
                                                    continue

                                                val_val_str_lower = str(
                                                    val
                                                ).lower()  # Convert value to lowercase string for comparison

                                                if val_val_str_lower in true_values:
                                                    result_val_series.loc[idx] = True
                                                elif val_val_str_lower in false_values:
                                                    result_val_series.loc[idx] = False
                                                else:
                                                    # This value is not recognized as bool and is not NaN
                                                    if (
                                                        len(unmappable_val_raw_values)
                                                        < 5
                                                    ):  # Collect a few examples of unmappable values
                                                        unmappable_val_raw_values.append(
                                                            val
                                                        )
                                                    conversion_val_possible = False  # Flag that not all values are mappable

                                        if (
                                            conversion_train_possible
                                            and conversion_test_possible
                                        ):
                                            df_train_processed[selected_column] = (
                                                result_train_series
                                            )
                                            df_test_processed[selected_column] = (
                                                result_test_series
                                            )

                                            if (
                                                st.session_state["validation_df"]
                                                is not None
                                            ):
                                                df_val_processed[selected_column] = (
                                                    result_val_series
                                                )

                                            message_type = "success"
                                            message_text = f":material/check_circle: **{selected_column}** column converted to **bool** successfully!"

                                        else:
                                            # Create a string list of unique example unmappable values for the message
                                            display_train_examples = list(
                                                pd.Series(unmappable_train_raw_values)
                                                .astype(str)
                                                .unique()
                                            )[:3]
                                            display_test_examples = list(
                                                pd.Series(unmappable_test_raw_values)
                                                .astype(str)
                                                .unique()
                                            )[:3]
                                            if (
                                                st.session_state["validation_df"]
                                                is not None
                                            ):
                                                display_val_examples = list(
                                                    pd.Series(unmappable_val_raw_values)
                                                    .astype(str)
                                                    .unique()
                                                )[:3]
                                            message_text = (
                                                f":material/cancel: Column **{selected_column}** cannot be reliably converted to **bool**. "
                                                f"It contains unmappable values. Examples: {display_train_examples}, {display_test_examples} & {display_val_examples}. "
                                                f"Please clean data or map values to True/False (or 1/0) first."
                                            )
                                            message_type = "error"
                                            st.session_state["message"] = (
                                                message_type,
                                                message_text,
                                            )

                                            if (
                                                st.session_state["validation_df"]
                                                is not None
                                            ):
                                                return (
                                                    df_train_input,
                                                    df_val_input,
                                                    df_test_input,
                                                )

                                            return (
                                                df_train_input,
                                                df_test_input,
                                            )  # Return original df

                                else:
                                    message_text = f":material/error_outline: Unsupported target datatype: **{target_datatype}**."
                                    message_type = "error"
                                    st.session_state["message"] = (
                                        message_type,
                                        message_text,
                                    )

                                    if st.session_state["validation_df"] is not None:
                                        return (
                                            df_train_input,
                                            df_val_input,
                                            df_test_input,
                                        )

                                    return df_train_input, df_test_input

                            except Exception as e:
                                # Return original values
                                df_train_processed[selected_column] = (
                                    original_train_column_series
                                )
                                df_test_processed[selected_column] = (
                                    original_test_column_series
                                )

                                # Error message
                                message_text = f":material/cancel: **Critical Error converting '{selected_column}' to {target_datatype}:** {str(e)}"
                                message_type = "error"
                                st.session_state["message"] = (
                                    message_type,
                                    message_text,
                                )

                                if st.session_state["validation_df"] is not None:
                                    df_val_processed[selected_column] = (
                                        original_val_column_series
                                    )
                                    return df_train_input, df_val_input, df_test_input

                                return df_train_input, df_test_input

                            # Update session_state
                            st.session_state["current_df"] = df_train_processed.copy()
                            st.session_state["test_df"] = df_test_processed.copy()

                            # Send message
                            st.session_state["message"] = (message_type, message_text)

                            if st.session_state["validation_df"] is not None:
                                return (
                                    df_train_processed,
                                    df_val_processed,
                                    df_test_processed,
                                )

                            return df_train_processed, df_test_processed

                        else:  # No split
                            df_processed = df_input.copy()
                            selected_column = params["selected_column"]
                            target_datatype = params["selected_datatype"]

                            # Check if the column exists
                            if selected_column not in df_processed.columns:
                                message_text = f":material/cancel: **Error:** Column **'{selected_column}'** not found in the DataFrame."
                                message_type = "error"
                                st.session_state["message"] = (
                                    message_type,
                                    message_text,
                                )

                                return df_input

                            original_column_series = df_processed[
                                selected_column
                            ].copy()  # For comparison and potential revert

                            try:
                                if target_datatype == "int":
                                    # Attempt to convert to numeric first (errors='coerce' turns non-numeric to NaN)
                                    numeric_series = pd.to_numeric(
                                        original_column_series, errors="coerce"
                                    )

                                    # Check if all values became NaN (and original wasn't all NaN)
                                    if (
                                        numeric_series.isnull().all()
                                        and not original_column_series.isnull().all()
                                    ):
                                        message_text = f":material/cancel: Could not convert any values in **{selected_column}** to numeric for integer conversion. All values resulted in NaN."
                                        message_type = "error"
                                        st.session_state["message"] = (
                                            message_type,
                                            message_text,
                                        )

                                        return df_input

                                    # Check for any NaNs. Standard 'int' (like np.int64) cannot have NaNs.
                                    if numeric_series.isnull().any():
                                        failed_conversions_count = (
                                            numeric_series.isnull()
                                            & original_column_series.notnull()
                                        ).sum()
                                        original_nans_count = (
                                            numeric_series.isnull()
                                            & original_column_series.isnull()
                                        ).sum()

                                        if failed_conversions_count > 0:
                                            message_text = f":material/warning: Column **{selected_column}** cannot be converted to **int**. {failed_conversions_count} value(s) were non-numeric and became NaN. "
                                        else:  # Only original NaNs are present
                                            message_text = f":material/warning: Column **{selected_column}** cannot be converted to **int** as it contains {original_nans_count} original NaN/NaT value(s) which are not supported by standard integer types. "
                                        message_type = "warning"
                                        st.session_state["message"] = (
                                            message_type,
                                            message_text,
                                        )

                                        return df_input  # Return original df as strict 'int' conversion failed

                                    else:
                                        # All values are numeric and non-null, safe to convert to standard int
                                        df_processed[selected_column] = (
                                            numeric_series.astype(np.int64)
                                        )
                                        message_type = "success"
                                        message_text = f":material/check_circle: **{selected_column}** column converted to **{target_datatype}** successfully!"

                                elif target_datatype == "float":
                                    numeric_series = pd.to_numeric(
                                        original_column_series, errors="coerce"
                                    )

                                    if (
                                        numeric_series.isnull().all()
                                        and not original_column_series.isnull().all()
                                    ):
                                        message_text = f":material/cancel: Could not convert any values in **{selected_column}** to **float**. All values resulted in NaN."
                                        message_type = "error"
                                        st.session_state["message"] = (
                                            message_type,
                                            message_text,
                                        )

                                        return df_input

                                    df_processed[selected_column] = (
                                        numeric_series.astype(np.float64)
                                    )
                                    new_nans_from_values = (
                                        numeric_series.isnull()
                                        & original_column_series.notnull()
                                    ).sum()

                                    if new_nans_from_values > 0:
                                        message_type = "warning"
                                        message_text = f":material/warning: **{selected_column}** column converted to **{target_datatype}**. However, {new_nans_from_values} value(s) could not be converted to numeric and are now NaN."
                                    else:
                                        message_type = "success"
                                        message_text = f":material/check_circle: **{selected_column}** column converted to **{target_datatype}** successfully!"

                                elif target_datatype == "datetime":
                                    datetime_series = pd.to_datetime(
                                        original_column_series, errors="coerce"
                                    )

                                    if (
                                        datetime_series.isnull().all()
                                        and not original_column_series.isnull().all()
                                    ):
                                        message_text = f":material/cancel: Could not convert any values in **{selected_column}** to **datetime**. All values resulted in NaT."
                                        message_type = "error"
                                        st.session_state["message"] = (
                                            message_type,
                                            message_text,
                                        )

                                        return df_input

                                    df_processed[selected_column] = datetime_series
                                    new_nats_from_values = (
                                        datetime_series.isnull()
                                        & original_column_series.notnull()
                                    ).sum()

                                    if new_nats_from_values > 0:
                                        message_type = "warning"
                                        message_text = (
                                            f":material/warning: **{selected_column}** column converted to **{target_datatype}**. "
                                            f"However, {new_nats_from_values} value(s) could not be converted to datetime and are now NaT."
                                        )
                                    else:
                                        message_type = "success"
                                        message_text = f":material/check_circle: **{selected_column}** column converted to **{target_datatype}** successfully!"

                                elif target_datatype == "category":
                                    num_unique = original_column_series.nunique()
                                    total_rows = len(original_column_series)

                                    # Heuristic thresholds for high cardinality
                                    high_cardinality_ratio = num_unique / total_rows
                                    high_cardinality = total_rows > 100 and (
                                        high_cardinality_ratio > 0.8 or num_unique > 200
                                    )

                                    if high_cardinality:
                                        message_text = (
                                            f" :material/info: Column **{selected_column}** has high cardinality "
                                            f"({num_unique} unique values out of {total_rows}, {high_cardinality_ratio:.1%}). "
                                            f"Converting to 'category' may not offer significant memory benefits or could slow down performance."
                                        )
                                        message_type = "info"
                                        st.session_state["message"] = (
                                            message_type,
                                            message_text,
                                        )
                                        return df_input

                                    else:
                                        df_processed[selected_column] = (
                                            original_column_series.astype("category")
                                        )
                                        message_type = "success"
                                        message_text = f":material/check_circle: **{selected_column}** column converted to **{target_datatype}** successfully!"

                                elif target_datatype == "str":
                                    df_processed[selected_column] = (
                                        original_column_series.astype(str)
                                    )
                                    message_type = "success"
                                    message_text = f":material/check_circle: **{selected_column}** column converted to **{target_datatype}** successfully!"

                                elif target_datatype == "bool":
                                    # Check if already a boolean type (including pandas nullable BooleanDtype)
                                    if pd.api.types.is_bool_dtype(
                                        original_column_series.dtype
                                    ):
                                        # If it's already bool, no conversion needed, but ensure df_processed reflects this.
                                        df_processed[selected_column] = (
                                            original_column_series
                                        )
                                        message_type = "success"
                                        message_text = f":material/check_circle: **{selected_column}** column is already **bool**."
                                    else:
                                        true_values = {
                                            "true",
                                            "1",
                                            "yes",
                                            "t",
                                            "y",
                                            "Yes",
                                            "Approved",
                                        }
                                        false_values = {
                                            "false",
                                            "0",
                                            "no",
                                            "f",
                                            "n",
                                            "No",
                                            "Rejected",
                                        }

                                        result_series = pd.Series(
                                            pd.NA,
                                            index=original_column_series.index,
                                            dtype=pd.BooleanDtype(),
                                        )
                                        unmappable_raw_values = []
                                        conversion_possible = True

                                        for idx, val in original_column_series.items():
                                            if pd.isna(val):  # Preserve original NaNs
                                                result_series.loc[idx] = pd.NA
                                                continue

                                            val_str_lower = str(
                                                val
                                            ).lower()  # Convert value to lowercase string for comparison

                                            if val_str_lower in true_values:
                                                result_series.loc[idx] = True
                                            elif val_str_lower in false_values:
                                                result_series.loc[idx] = False
                                            else:
                                                # This value is not recognized as bool and is not NaN
                                                if (
                                                    len(unmappable_raw_values) < 5
                                                ):  # Collect a few examples of unmappable values
                                                    unmappable_raw_values.append(val)
                                                conversion_possible = False  # Flag that not all values are mappable

                                        if conversion_possible:
                                            df_processed[selected_column] = (
                                                result_series
                                            )
                                            message_type = "success"
                                            message_text = f":material/check_circle: **{selected_column}** column converted to **bool** successfully!"
                                        else:
                                            # Create a string list of unique example unmappable values for the message
                                            display_examples = list(
                                                pd.Series(unmappable_raw_values)
                                                .astype(str)
                                                .unique()
                                            )[:3]
                                            message_text = (
                                                f":material/cancel: Column **{selected_column}** cannot be reliably converted to **bool**. "
                                                f"It contains unmappable values. Examples: {display_examples}. "
                                                f"Please clean data or map values to True/False (or 1/0) first."
                                            )
                                            message_type = "error"
                                            st.session_state["message"] = (
                                                message_type,
                                                message_text,
                                            )

                                            return df_input

                                else:
                                    message_text = f":material/error_outline: Unsupported target datatype: **{target_datatype}**."
                                    message_type = "error"
                                    st.session_state["message"] = (
                                        message_type,
                                        message_text,
                                    )

                                    return df_input

                            except Exception as e:
                                df_processed[selected_column] = original_column_series
                                message_text = f":material/cancel: **Critical Error converting '{selected_column}' to {target_datatype}:** {str(e)}"
                                message_type = "error"
                                st.session_state["message"] = (
                                    message_type,
                                    message_text,
                                )
                                return df_input

                            # Update session_state
                            st.session_state["current_df"] = df_processed.copy()

                            # Send message
                            st.session_state["message"] = (message_type, message_text)

                            return df_processed

                    case "train_test_split":
                        test_size = params.get("test_size", 0.2)
                        random_state = params.get("random_state", 42)
                        shuffle = params.get("shuffle", True)
                        stratify_col = params.get("stratify_col")
                        num_bins = params.get("stratify_num_bins", 10)

                        st.session_state.is_data_split = False
                        stratify_data, stratify_msg = None, ""

                        if stratify_col and stratify_col in df_input.columns:
                            col_data = df_input[stratify_col]
                            is_numeric = pd.api.types.is_numeric_dtype(col_data)
                            unique_vals = col_data.nunique()
                            needs_binning = is_numeric and unique_vals > max(
                                num_bins * 2, 25
                            )

                            if needs_binning:
                                st.info(
                                    f"Numerical column '{stratify_col}' has {unique_vals} unique values. Binning into {num_bins} bins.",
                                    icon=":material/info:",
                                )
                                try:
                                    stratify_data = pd.qcut(
                                        col_data,
                                        q=num_bins,
                                        labels=False,
                                        duplicates="drop",
                                    )
                                    stratify_msg = f", stratified by binned '{stratify_col}' ({stratify_data.nunique()} bins)"
                                except Exception as e:
                                    st.warning(
                                        f":material/warning: Could not bin column '{stratify_col}' for stratification ({e}). Proceeding without stratification."
                                    )
                                    stratify_msg = ", stratification skipped due to binning failure"
                            else:
                                stratify_data = col_data
                                stratify_msg = f", stratified by '{stratify_col}'"

                        try:
                            df_train, df_test = train_test_split(
                                df_input,
                                test_size=test_size,
                                random_state=random_state,
                                shuffle=shuffle,
                                stratify=stratify_data,
                            )

                            st.session_state.update(
                                {
                                    "train_df": df_train.copy(),
                                    "initial_train_df": df_train.copy(),
                                    "test_df": df_test.copy(),
                                    "initial_test_df": df_test.copy(),
                                    "current_df": df_train.copy(),
                                    "is_data_split": True,
                                    "message": (
                                        "success",
                                        f":material/check_circle: Data split into train and test sets (test size = **{test_size*100:.0f}%**){stratify_msg}",
                                    ),
                                }
                            )

                            return df_train, df_test

                        except ValueError as e:
                            st.session_state["message"] = (
                                "error",
                                f":material/error: Data split failed: {e}. Possibly due to small stratification groups.",
                            )
                            return None, None

                    case "train_test_val_split":

                        def error_message(text):
                            st.session_state["message"] = (
                                "error",
                                f":material/error: {text}",
                            )
                            st.session_state.is_data_split = False
                            clear_session_dfs()
                            return None, None, None

                        def clear_session_dfs():
                            for key in [
                                "train_df",
                                "initial_train_df",
                                "validation_df",
                                "initial_validation_df",
                                "test_df",
                                "initial_test_df",
                                "current_df",
                            ]:
                                st.session_state.pop(key, None)

                        def validate_split_sizes(test, val):
                            if not (0 < test < 1):
                                return error_message(
                                    f"Test size must be between 0 and 1, got {test}."
                                )
                            if not (0 < val < 1):
                                return error_message(
                                    f"Validation size must be between 0 and 1, got {val}."
                                )
                            if test + val >= 1.0:
                                return error_message(
                                    f"Sum of test_size ({test:.2f}) and validation_size ({val:.2f}) must be < 1.0."
                                )
                            return None

                        def prepare_stratify_data(df, col, num_bins):
                            suffix, stratify_data = "", None
                            if col and col in df.columns:
                                col_data = df[col]
                                unique_vals = col_data.nunique()
                                if pd.api.types.is_numeric_dtype(
                                    col_data
                                ) and unique_vals > max(num_bins * 2, 25):
                                    try:
                                        stratify_data = pd.qcut(
                                            col_data,
                                            q=num_bins,
                                            labels=False,
                                            duplicates="drop",
                                        )
                                        suffix = f", stratified by binned '{col}' ({stratify_data.nunique()} bins)"
                                        if stratify_data.nunique() < 2:
                                            st.warning(
                                                f":material/warning: Binning column '{col}' has only {stratify_data.nunique()} bin(s)."
                                            )
                                    except Exception as e:
                                        st.warning(
                                            f":material/warning: Could not bin column '{col}' for stratification ({e})."
                                        )
                                        suffix = f", stratification by '{col}' skipped due to binning failure"
                                else:
                                    stratify_data = col_data
                                    if (
                                        col_data.nunique() > 0
                                        and col_data.value_counts().min() < 2
                                    ):
                                        st.warning(
                                            f":material/warning: Column '{col}' has groups with too few samples."
                                        )
                                    suffix = f", stratified by '{col}'"
                            elif col:
                                st.warning(
                                    f":material/warning: Stratification column '{col}' not found."
                                )
                                suffix = ", stratification column not found"
                            return stratify_data, suffix

                        # --- Get Parameters ---
                        test_size = params.get("test_size", 0.2)
                        validation_size = params.get("validation_size", 0.1)
                        random_state = params.get("random_state", 42)
                        shuffle = params.get("shuffle", True)
                        stratify_col = params.get("stratify_col")
                        num_bins = params.get("stratify_num_bins", 10)

                        st.session_state.is_data_split = False

                        # --- Validate Sizes ---
                        if err := validate_split_sizes(test_size, validation_size):
                            return err

                        # --- Prepare Stratification ---
                        stratify_data, strat_suffix = prepare_stratify_data(
                            df_input, stratify_col, num_bins
                        )
                        strat1 = (
                            stratify_data
                            if stratify_col and stratify_data is not None
                            else None
                        )

                        try:
                            # First split: train+val / test
                            df_train_val, df_test = train_test_split(
                                df_input,
                                test_size=test_size,
                                random_state=random_state,
                                shuffle=shuffle,
                                stratify=strat1,
                            )

                            rel_val_size = validation_size / (1 - test_size)
                            if not (0 < rel_val_size < 1):
                                return error_message(
                                    f"Invalid relative validation size ({rel_val_size:.2f})."
                                )

                            strat2 = (
                                stratify_data.loc[df_train_val.index]
                                if strat1 is not None
                                else None
                            )
                            df_train, df_validation = train_test_split(
                                df_train_val,
                                test_size=rel_val_size,
                                random_state=random_state,
                                shuffle=shuffle,
                                stratify=strat2,
                            )

                            # Store in session state
                            for name, df in zip(
                                ["train", "validation", "test"],
                                [df_train, df_validation, df_test],
                            ):
                                st.session_state[f"{name}_df"] = df.copy()
                                st.session_state[f"initial_{name}_df"] = df.copy()
                            st.session_state["current_df"] = df_train.copy()
                            st.session_state.is_data_split = True

                            # Success message
                            total = len(df_input)
                            msg = (
                                f":material/check_circle: Data split into train ({len(df_train)/total:.1%}), "
                                f"validation ({len(df_validation)/total:.1%}), and test ({len(df_test)/total:.1%}) sets{strat_suffix}."
                            )
                            st.session_state["message"] = ("success", msg)

                            return df_train, df_validation, df_test

                        except ValueError as e:
                            return error_message(f"Data split failed: {e}")
                        except Exception as e:
                            return error_message(f"Unexpected error during split: {e}")

                    case "remove_outliers":
                        if is_split:
                            # Get copies of the current train and test data from session state.
                            col = params["selected_outlier_column"]

                            df_train_input = df_input
                            df_test_input = st.session_state["test_df"].copy()

                            df_train_processed = df_train_input
                            df_test_processed = df_test_input

                            if st.session_state["validation_df"] is not None:
                                df_validation_input = st.session_state[
                                    "validation_df"
                                ].copy()
                                df_validation_processed = df_validation_input

                            match params["outlier_removal_method"]:
                                case "IQR":
                                    q1 = df_train_input[col].quantile(0.25)
                                    q3 = df_train_input[col].quantile(0.75)
                                    iqr = q3 - q1
                                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

                                    before_train = len(df_train_input)
                                    df_train_processed = df_train_input[
                                        (df_train_input[col] >= lower)
                                        & (df_train_input[col] <= upper)
                                    ]
                                    after_train = len(df_train_processed)

                                    before_test = len(df_test_input)
                                    df_test_processed = df_test_input[
                                        (df_test_input[col] >= lower)
                                        & (df_test_input[col] <= upper)
                                    ]
                                    after_test = len(df_test_processed)

                                    if st.session_state["validation_df"] is not None:
                                        before_val = len(df_validation_input)
                                        df_validation_processed = df_validation_input[
                                            (df_validation_input[col] >= lower)
                                            & (df_validation_input[col] <= upper)
                                        ]
                                        after_val = len(df_validation_processed)

                                    message_type = "success"
                                    if st.session_state["validation_df"] is not None:
                                        message_text = f":material/check_circle: Removed **{before_train - after_train}** outliers from train, **{before_val - after_val}** from validation & **{before_test - after_test}** from test (IQR)."
                                    else:
                                        message_text = f":material/check_circle: Removed **{before_train - after_train}** outliers from train, **{before_test - after_test}** from test (IQR)."

                                case "Percentage":
                                    lower = df_train_input[col].quantile(
                                        params["bottom_percentage"]
                                    )
                                    upper = df_train_input[col].quantile(
                                        1 - params["top_percentage"]
                                    )

                                    before_train = len(df_train_input)
                                    df_train_processed = df_train_input[
                                        (df_train_input[col] >= lower)
                                        & (df_train_input[col] <= upper)
                                    ]
                                    after_train = len(df_train_processed)

                                    before_test = len(df_test_input)
                                    df_test_processed = df_test_input[
                                        (df_test_input[col] >= lower)
                                        & (df_test_input[col] <= upper)
                                    ]
                                    after_test = len(df_test_processed)

                                    if st.session_state["validation_df"] is not None:
                                        before_val = len(df_validation_input)
                                        df_validation_processed = df_validation_input[
                                            (df_validation_input[col] >= lower)
                                            & (df_validation_input[col] <= upper)
                                        ]
                                        after_val = len(df_validation_processed)

                                    message_type = "success"
                                    if st.session_state["validation_df"] is not None:
                                        message_text = f":material/check_circle: Removed **{before_train - after_train}** from train, **{before_val - after_val}** from validation & **{before_test - after_test}** from test (Percentage)."
                                    else:
                                        message_text = f":material/check_circle: Removed **{before_train - after_train}** from train, **{before_test - after_test}** from test (Percentage)."

                                case "Isolation Forest":
                                    model = IsolationForest(
                                        contamination=0.05, random_state=42
                                    )
                                    model.fit(df_train_input[[col]])

                                    before_train = len(df_train_input)
                                    preds_train = model.predict(df_train_input[[col]])
                                    df_train_processed = df_train_input[
                                        preds_train == 1
                                    ]
                                    after_train = len(df_train_processed)

                                    before_test = len(df_test_input)
                                    preds_test = model.predict(df_test_input[[col]])
                                    df_test_processed = df_test_input[preds_test == 1]
                                    after_test = len(df_test_processed)

                                    if st.session_state["validation_df"] is not None:
                                        before_val = len(df_validation_input)
                                        preds_val = model.predict(
                                            df_validation_input[[col]]
                                        )
                                        df_validation_processed = df_validation_input[
                                            preds_val == 1
                                        ]
                                        after_val = len(df_validation_processed)

                                    message_type = "success"
                                    if st.session_state["validation_df"] is not None:
                                        message_text = f":material/check_circle: Removed **{before_train - after_train}** from train, **{before_val - after_val}** from validation & **{before_test - after_test}** from test (Isolation Forest)."
                                    else:
                                        message_text = f":material/check_circle: Removed **{before_train - after_train}** from train, **{before_test - after_test}** from test (Isolation Forest)."

                            #  Update session state with copies of the *processed* DataFrames.
                            st.session_state["current_df"] = df_train_processed.copy()
                            st.session_state["test_df"] = df_test_processed.copy()

                            # OPTIONAL
                            if (
                                "train_df" in st.session_state
                                and st.session_state["train_df"] is not None
                            ):
                                st.session_state["train_df"] = df_train_processed.copy()

                            # Send message
                            st.session_state["message"] = (message_type, message_text)

                            if st.session_state["validation_df"] is not None:
                                st.session_state["validation_df"] = (
                                    df_validation_processed.copy()
                                )
                                return (
                                    df_train_processed,
                                    df_validation_processed,
                                    df_test_processed,
                                )

                            return df_train_processed, df_test_processed

                        else:  # Not split
                            col = params["selected_outlier_column"]
                            df_processed = df_input.copy()

                            match params["outlier_removal_method"]:
                                case "IQR":
                                    q1 = df_input[col].quantile(0.25)
                                    q3 = df_input[col].quantile(0.75)
                                    iqr = q3 - q1
                                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                                    before = len(df_input)
                                    df_processed = df_input[
                                        (df_input[col] >= lower)
                                        & (df_input[col] <= upper)
                                    ]
                                    after = len(df_processed)
                                    message_type = "success"
                                    message_text = f":material/check_circle: Removed **{before - after}** outliers from column **{col}** using IQR method."

                                case "Percentage":
                                    lower = df_input[col].quantile(
                                        params["bottom_percentage"]
                                    )
                                    upper = df_input[col].quantile(
                                        1 - params["top_percentage"]
                                    )
                                    df_processed = df_input[
                                        (df_input[col] >= lower)
                                        & (df_input[col] <= upper)
                                    ]
                                    message_type = "success"
                                    message_text = f":material/check_circle: Removed top **{params['top_percentage']*100:.2f}%** and bottom **{params['bottom_percentage']*100:.2f}%** outliers from column **{col}**."

                                case "Isolation Forest":
                                    before = len(df_input)
                                    model = IsolationForest(
                                        contamination=0.05, random_state=42
                                    )
                                    preds = model.fit_predict(df_input[[col]])
                                    df_processed = df_input[preds == 1]
                                    after = len(df_processed)
                                    message_type = "success"
                                    message_text = f":material/check_circle: Removed **{before - after}** outliers from column **{col}** using Isolation Forest method."

                            # Update session_state
                            st.session_state["current_df"] = df_processed

                            # Send message
                            st.session_state["message"] = (message_type, message_text)

                            return df_processed

                    case "scale_columns":
                        # Set the scaling flag to False
                        st.session_state.is_data_scaled = False

                        if is_split:
                            scaling_method = params["scaling_method"]

                            # Get copies of the current train and test data from session state.
                            df_train_input = df_input
                            df_test_input = st.session_state["test_df"].copy()

                            # Ensure numeric columns are identified correctly for potentially already modified DFs
                            num_cols_train = df_train_input.select_dtypes(
                                include="number"
                            ).columns
                            num_cols_test = df_test_input.select_dtypes(
                                include="number"
                            ).columns

                            df_train_processed = df_train_input
                            df_test_processed = df_test_input

                            scaler_instance = None
                            if scaling_method == "StandardScaler (Z-score)":
                                scaler_instance = StandardScaler()
                            elif scaling_method == "MinMaxScaler (0-1 range)":
                                scaler_instance = MinMaxScaler()
                            elif scaling_method == "Robust Scaler":
                                scaler_instance = RobustScaler()

                            if (
                                scaler_instance
                                and not df_train_processed[num_cols_train].empty
                            ):
                                df_train_processed[num_cols_train] = (
                                    scaler_instance.fit_transform(
                                        df_train_processed[num_cols_train]
                                    )
                                )
                                if not df_test_processed[
                                    num_cols_test
                                ].empty:  # Ensure test set also has these numeric columns
                                    # Check if num_cols_test has columns, can happen if test set becomes empty or loses numeric cols
                                    if len(num_cols_test) > 0 and all(
                                        col in df_test_processed.columns
                                        for col in num_cols_test
                                    ):
                                        df_test_processed[num_cols_test] = (
                                            scaler_instance.transform(
                                                df_test_processed[num_cols_test]
                                            )
                                        )
                                    elif (
                                        len(num_cols_test) == 0
                                        and len(num_cols_train) > 0
                                    ):
                                        message_type = "warning"
                                        message_text = "Test set had no numeric columns to scale, while train set did."

                                    if st.session_state["validation_df"] is not None:
                                        df_val_input = st.session_state[
                                            "validation_df"
                                        ].copy()
                                        df_val_processed = df_val_input
                                        num_cols_val = df_val_input.select_dtypes(
                                            include="number"
                                        ).columns

                                        # Transform validation set
                                        df_val_processed[num_cols_val] = (
                                            scaler_instance.transform(
                                                df_val_processed[num_cols_val]
                                            )
                                        )

                                        # Update session state
                                        st.session_state["validation_df"] = (
                                            df_val_processed.copy()
                                        )

                            # Update session state.
                            st.session_state["current_df"] = df_train_processed.copy()
                            st.session_state["test_df"] = df_test_processed.copy()

                            if (
                                "train_df" in st.session_state
                                and st.session_state["train_df"] is not None
                            ):
                                st.session_state["train_df"] = df_train_processed.copy()

                            # Set scaling flag to True
                            st.session_state.is_data_scaled = True

                            # Send message
                            message_type = "success"
                            message_text = f":material/check_circle: Columns scaled with {scaling_method} successfully!"
                            st.session_state["message"] = (message_type, message_text)

                            if st.session_state["validation_df"] is not None:
                                return (
                                    df_train_processed,
                                    df_val_processed,
                                    df_test_processed,
                                )

                            return df_train_processed, df_test_processed

                        else:  # No split
                            scaling_method = params["scaling_method"]
                            num_cols = df_input.select_dtypes(include="number").columns
                            df_processed = df_input

                            match scaling_method:
                                case "StandardScaler (Z-score)":
                                    df_processed[
                                        num_cols
                                    ] = StandardScaler().fit_transform(
                                        df_processed[num_cols]
                                    )
                                    message_text = f":material/check_circle: Columns scaled with {scaling_method} successfully!"
                                case "MinMaxScaler (0-1 range)":
                                    df_processed[
                                        num_cols
                                    ] = MinMaxScaler().fit_transform(
                                        df_processed[num_cols]
                                    )
                                    message_text = f":material/check_circle: Columns scaled with {scaling_method} successfully!"
                                case "Robust Scaler":
                                    df_processed[
                                        num_cols
                                    ] = RobustScaler().fit_transform(
                                        df_processed[num_cols]
                                    )
                                    message_text = f":material/check_circle: Columns scaled with {scaling_method} successfully!"

                            # Set scaling flag to True
                            st.session_state.is_data_scaled = True

                            # Send message
                            message_type = "success"
                            st.session_state["message"] = (message_type, message_text)

                            return df_processed

                    case "encode_columns":
                        # Set initial flag to false
                        st.session_state.is_data_encoded = False

                        if is_split:
                            # Get copies of the current train and test data from session state.
                            df_train_input = df_input
                            df_test_input = st.session_state["test_df"].copy()

                            df_train_processed = df_train_input.copy()
                            df_test_processed = df_test_input.copy()

                            if st.session_state["validation_df"] is not None:
                                df_val_input = st.session_state["validation_df"].copy()
                                df_val_processed = df_val_input.copy()

                            # Get the list of categorical columns intended for encoding from parameters
                            candidate_cat_cols = params.get("cat_cols", [])

                            # Filter to ensure these columns still exist in the current train DataFrame and are 'object' type
                            actual_cat_cols_to_encode = [
                                col
                                for col in candidate_cat_cols
                                if col in df_train_processed.columns
                                and df_train_processed[col].dtype == "object"
                            ]

                            if not actual_cat_cols_to_encode:
                                message_text = ":material/warning: No suitable categorical columns found in the current training data for encoding."
                                message_type = "warning"
                                st.session_state["message"] = (
                                    message_type,
                                    message_text,
                                )

                                # Return the DFs unmodified, but ensure session state is also "updated" with these current (unmodified) versions
                                st.session_state["current_df"] = df_train_input.copy()
                                st.session_state["test_df"] = df_test_input.copy()

                                # OPTIONAL
                                if (
                                    "train_df" in st.session_state
                                    and st.session_state["train_df"] is not None
                                ):
                                    st.session_state["train_df"] = df_train_input.copy()

                                if st.session_state["validation_df"] is not None:
                                    st.session_state["validation_df"] = (
                                        df_val_input.copy()
                                    )
                                    return df_train_input, df_val_input, df_test_input

                                return (
                                    df_train_input,
                                    df_test_input,
                                )  # Return unmodified DFs

                            encoding_method = params["encoding_method"]

                            match encoding_method:
                                case "Label Encoding":
                                    # It might fail if there are "never seen before" labels in test or val sets
                                    # Use Ordinal Encoder instead
                                    for col in actual_cat_cols_to_encode:
                                        le = LabelEncoder()
                                        df_train_processed[col] = le.fit_transform(
                                            df_train_processed[col].astype(str)
                                        )
                                        # Ensure the column exists in test set before transforming
                                        if col in df_test_processed.columns:
                                            df_test_processed[col] = le.transform(
                                                df_test_processed[col].astype(str)
                                            )
                                        else:
                                            st.warning(
                                                f"Column {col} not found in test set for Label Encoding, skipping.",
                                                icon=":material/warning:",
                                            )

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            df_val_processed[col] = le.transform(
                                                df_val_processed[col].astype(str)
                                            )

                                    message_type = "success"
                                    message_text = f":material/check_circle: Applied Label Encoding successfully to: {', '.join(actual_cat_cols_to_encode)}!"

                                case "Ordinal Encoding":
                                    encoder = OrdinalEncoder(
                                        handle_unknown="use_encoded_value",
                                        unknown_value=-1,
                                    )
                                    df_train_processed[actual_cat_cols_to_encode] = (
                                        encoder.fit_transform(
                                            df_train_processed[
                                                actual_cat_cols_to_encode
                                            ].astype(str)
                                        )
                                    )

                                    if df_test_processed is not None:
                                        df_test_processed[actual_cat_cols_to_encode] = (
                                            encoder.transform(
                                                df_test_processed[
                                                    actual_cat_cols_to_encode
                                                ].astype(str)
                                            )
                                        )

                                    if st.session_state["validation_df"] is not None:
                                        df_val_processed[actual_cat_cols_to_encode] = (
                                            encoder.transform(
                                                df_val_processed[
                                                    actual_cat_cols_to_encode
                                                ].astype(str)
                                            )
                                        )

                                    message_type = "success"
                                    message_text = f":material/check_circle: Applied Ordinal Encoding successfully to: {', '.join(actual_cat_cols_to_encode)}!"

                                case "One-Hot Encoding":
                                    # Define columns that are not being one-hot encoded
                                    remainder_cols = [
                                        col
                                        for col in df_train_processed.columns
                                        if col not in actual_cat_cols_to_encode
                                    ]

                                    preprocessor = ColumnTransformer(
                                        transformers=[
                                            (
                                                "cat",
                                                OneHotEncoder(
                                                    drop="first",
                                                    sparse_output=False,
                                                    handle_unknown="ignore",
                                                ),
                                                actual_cat_cols_to_encode,
                                            )
                                        ],
                                        remainder="passthrough",  # Keep other columns
                                    )

                                    # Fit on training data
                                    preprocessor.fit(df_train_processed)

                                    # Get new column names after OHE
                                    ohe_feature_names = (
                                        preprocessor.named_transformers_["cat"]
                                        .get_feature_names_out(
                                            actual_cat_cols_to_encode
                                        )
                                        .tolist()
                                    )
                                    new_col_names = ohe_feature_names + remainder_cols

                                    # Transform train data
                                    train_encoded_data = preprocessor.transform(
                                        df_train_processed
                                    )
                                    df_train_processed = pd.DataFrame(
                                        train_encoded_data,
                                        columns=new_col_names,
                                        index=df_train_input.index,
                                    )

                                    # Transform test data
                                    if not df_test_processed.empty:
                                        test_encoded_data = preprocessor.transform(
                                            df_test_processed
                                        )
                                        df_test_processed = pd.DataFrame(
                                            test_encoded_data,
                                            columns=new_col_names,
                                            index=df_test_input.index,
                                        )

                                    # Transform validation data
                                    if st.session_state["validation_df"] is not None:
                                        val_encoded_data = preprocessor.transform(
                                            df_val_processed
                                        )
                                        df_val_processed = pd.DataFrame(
                                            val_encoded_data,
                                            columns=new_col_names,
                                            index=df_val_input.index,
                                        )

                                    # Reorder the DataFrame columns
                                    final_train_column_order = (
                                        remainder_cols + ohe_feature_names
                                    )
                                    final_test_column_order = (
                                        remainder_cols + ohe_feature_names
                                    )

                                    # Apply the desired column order to the DataFrame
                                    df_train_processed = df_train_processed[
                                        final_train_column_order
                                    ]
                                    df_test_processed = df_test_processed[
                                        final_test_column_order
                                    ]

                                    if st.session_state["validation_df"] is not None:
                                        final_val_column_order = (
                                            remainder_cols + ohe_feature_names
                                        )
                                        df_val_processed = df_val_processed[
                                            final_val_column_order
                                        ]

                                    message_type = "success"
                                    message_text = f":material/check_circle: Applied One-Hot Encoding successfully to: {', '.join(actual_cat_cols_to_encode)}!"

                                case "Target Encoding":
                                    target_column = params.get("target_column")
                                    if (
                                        not target_column
                                        or target_column
                                        not in df_train_processed.columns
                                    ):
                                        message_text = f":material/error: Target column '{target_column}' not found for Target Encoding."
                                        message_type = "error"
                                        st.session_state["message"] = (
                                            message_type,
                                            message_text,
                                        )
                                        # Return DFs unmodified from input of this operation
                                        st.session_state["current_df"] = (
                                            df_train_input.copy()
                                        )
                                        st.session_state["test_df"] = (
                                            df_test_input.copy()
                                        )
                                        if (
                                            "train_df" in st.session_state
                                            and st.session_state["train_df"] is not None
                                        ):
                                            st.session_state["train_df"] = (
                                                df_train_input.copy()
                                            )

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            st.session_state["validation_df"] = (
                                                df_val_input.copy()
                                            )

                                            return (
                                                df_train_input,
                                                df_val_input,
                                                df_test_input,
                                            )

                                        return df_train_input, df_test_input

                                    for col in actual_cat_cols_to_encode:
                                        if col == target_column:
                                            continue  # Skip if categorical column is the target itself

                                        encoded_col_name = f"{col}_TargetEncoded"
                                        means = df_train_processed.groupby(col)[
                                            target_column
                                        ].mean()

                                        df_train_processed[encoded_col_name] = (
                                            df_train_processed[col].map(means)
                                        )

                                        if col in df_test_processed.columns:
                                            df_test_processed[encoded_col_name] = (
                                                df_test_processed[col].map(means)
                                            )

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            df_val_processed[encoded_col_name] = (
                                                df_val_processed[col].map(means)
                                            )

                                        # Fill NaNs that might appear (e.g., category in test not in train, or if a group mean was NaN)
                                        overall_mean = df_train_processed[
                                            target_column
                                        ].mean()  # Use overall mean from training set
                                        df_train_processed[encoded_col_name].fillna(
                                            overall_mean, inplace=True
                                        )
                                        if (
                                            encoded_col_name
                                            in df_test_processed.columns
                                        ):  # Check if column was added
                                            df_test_processed[encoded_col_name].fillna(
                                                overall_mean, inplace=True
                                            )

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            if (
                                                encoded_col_name
                                                in df_val_processed.columns
                                            ):  # Check if column was added
                                                df_val_processed[
                                                    encoded_col_name
                                                ].fillna(overall_mean, inplace=True)

                                    message_type = "success"
                                    message_text = f":material/check_circle: Applied Target Encoding against '{target_column}' for columns: {', '.join(actual_cat_cols_to_encode)} successfully!"

                            # Update session state with copies of the *processed* DataFrames.
                            st.session_state["current_df"] = df_train_processed.copy()
                            st.session_state["test_df"] = df_test_processed.copy()
                            if (
                                "train_df" in st.session_state
                                and st.session_state["train_df"] is not None
                            ):
                                st.session_state["train_df"] = df_train_processed.copy()

                            if st.session_state["validation_df"] is not None:
                                st.session_state["validation_df"] = (
                                    df_val_processed.copy()
                                )

                            if message_type == "success":
                                st.session_state.is_data_encoded = True
                            st.session_state["message"] = (message_type, message_text)

                            if st.session_state["validation_df"] is not None:
                                return (
                                    df_train_processed,
                                    df_val_processed,
                                    df_test_processed,
                                )

                            return df_train_processed, df_test_processed

                        else:  # No split
                            cat_cols = df_input.select_dtypes(include="object").columns
                            num_cols = df_input.select_dtypes(
                                include=["number"]
                            ).columns
                            df_processed = df_input

                            # Filter to ensure these columns still exist in the current train DataFrame and are 'object' type
                            actual_cat_cols_to_encode = [
                                col
                                for col in cat_cols
                                if col in df_processed.columns
                                and df_processed[col].dtype == "object"
                            ]

                            encoding_method = params["encoding_method"]
                            match encoding_method:
                                case "Label Encoding":

                                    for col in cat_cols:
                                        le = LabelEncoder()
                                        df_processed[col] = le.fit_transform(
                                            df_processed[col].astype(str)
                                        )
                                        message_type = "success"
                                        message_text = f":material/check_circle: Applied {encoding_method} to categorical variables successfully!"

                                case "Ordinal Encoding":
                                    encoder = OrdinalEncoder(
                                        handle_unknown="use_encoded_value",
                                        unknown_value=-1,
                                    )
                                    df_processed[cat_cols] = encoder.fit_transform(
                                        df_processed[cat_cols].astype(str)
                                    )

                                    message_type = "success"
                                    message_text = f":material/check_circle: Applied {encoding_method} to categorical variables successfully!"

                                case "One-Hot Encoding":
                                    remainder_cols = [
                                        col
                                        for col in df_processed.columns
                                        if col not in actual_cat_cols_to_encode
                                    ]

                                    preprocessor = ColumnTransformer(
                                        transformers=[
                                            (
                                                "cat",
                                                OneHotEncoder(
                                                    drop="first",
                                                    sparse_output=False,
                                                    handle_unknown="ignore",
                                                ),
                                                actual_cat_cols_to_encode,
                                            )
                                        ],
                                        remainder="passthrough",  # Keep other columns
                                    )

                                    # Fit and transform the data
                                    encoded_data = preprocessor.fit_transform(
                                        df_processed
                                    )

                                    # Get the names of the newly created OHE columns from the fitted transformer
                                    ohe_feature_names = (
                                        preprocessor.named_transformers_["cat"]
                                        .get_feature_names_out(
                                            actual_cat_cols_to_encode
                                        )
                                        .tolist()
                                    )

                                    columns_in_transformer_order = (
                                        ohe_feature_names + remainder_cols
                                    )

                                    # Create the DataFrame with columns in the transformer's default order
                                    # Use the index from the original df_processed to preserve it
                                    df_processed = pd.DataFrame(
                                        encoded_data,
                                        columns=columns_in_transformer_order,
                                        index=df_processed.index,
                                    )

                                    # Reorder the DataFrame columns
                                    final_column_order = (
                                        remainder_cols + ohe_feature_names
                                    )

                                    # Apply the desired column order to the DataFrame
                                    df_processed = df_processed[final_column_order]

                                    message_type = "success"
                                    message_text = f":material/check_circle: Applied {encoding_method} to categorical variables successfully!"

                                case "Target Encoding":
                                    target_column = params["target_column"]
                                    for col in cat_cols:
                                        new_col_name = f"{col}_TargetEncoded"
                                        df_processed[new_col_name] = target_encode(
                                            df_processed,
                                            col,
                                            target_column,
                                            smoothing=0.0,
                                        )
                                        message_type = "success"
                                        message_text = f":material/check_circle: Applied {encoding_method} against {target_column} column successfully!"

                            # Update session_state
                            st.session_state["current_df"] = df_processed.copy()

                            # Set encoding flag to True
                            if message_type == "success":
                                st.session_state.is_data_encoded = True

                            # Send messge
                            st.session_state["message"] = (message_type, message_text)

                            return df_processed

                    case "transform_columns":
                        # Set initial flag to false
                        st.session_state.is_data_transformed = False

                        if is_split:
                            # Get copies of the current train and test data from session state.
                            df_train_input = st.session_state["current_df"].copy()
                            df_test_input = st.session_state["test_df"].copy()

                            df_train_processed = df_train_input.copy()
                            df_test_processed = df_test_input.copy()

                            if st.session_state["validation_df"] is not None:
                                df_val_input = st.session_state["validation_df"].copy()
                                df_val_processed = df_val_input.copy()

                            col_to_transform = params["col_to_transform"]
                            transform_method = params["transform_method"]

                            # Ensure the column exists in the dataframes
                            if col_to_transform not in df_train_processed.columns:
                                message_text = f":material/error: Column '{col_to_transform}' not found in training data for transformation."
                                message_type = "error"
                                st.session_state["message"] = (
                                    message_type,
                                    message_text,
                                )
                                # Return DFs unmodified from input of this operation
                                st.session_state["current_df"] = df_train_input.copy()
                                st.session_state["test_df"] = df_test_input.copy()
                                if (
                                    "train_df" in st.session_state
                                    and st.session_state["train_df"] is not None
                                ):
                                    st.session_state["train_df"] = df_train_input.copy()

                                if st.session_state["validation_df"] is not None:
                                    st.session_state["validation_df"] = (
                                        df_val_input.copy()
                                    )
                                    return df_train_input, df_val_input, df_test_input

                                return df_train_input, df_test_input

                            original_message_type = "success"

                            match transform_method:
                                case "Log Transform":
                                    if (
                                        (df_train_processed[col_to_transform] < 0)
                                        .any()
                                        .any()
                                    ):  # Check on train_processed
                                        message_text = f":material/warning: Log transform (log1p) skipped for column '{col_to_transform}' due to negative values in training data."
                                        original_message_type = "warning"
                                    else:
                                        df_train_processed[col_to_transform] = np.log1p(
                                            df_train_processed[col_to_transform]
                                        )
                                        if (
                                            col_to_transform
                                            in df_test_processed.columns
                                        ):
                                            df_test_processed[col_to_transform] = (
                                                np.log1p(
                                                    df_test_processed[col_to_transform]
                                                )
                                            )

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            df_val_processed[col_to_transform] = (
                                                np.log1p(
                                                    df_val_processed[col_to_transform]
                                                )
                                            )

                                        message_text = f":material/check_circle: Log transform (log1p) applied to '{col_to_transform}' successfully!"

                                case "Box-Cox Transform":
                                    if (
                                        (df_train_processed[col_to_transform] <= 0)
                                        .any()
                                        .any()
                                    ):  # Check on train_processed
                                        message_text = f":material/warning: Box-Cox transform skipped for column '{col_to_transform}' as training data contains non-positive values."
                                        original_message_type = "warning"
                                    else:
                                        try:
                                            transformed_train_data, lambda_value = (
                                                stats.boxcox(
                                                    df_train_processed[col_to_transform]
                                                )
                                            )
                                            df_train_processed[col_to_transform] = (
                                                transformed_train_data
                                            )

                                            if (
                                                col_to_transform
                                                in df_test_processed.columns
                                            ):
                                                # Ensure test data is also positive for Box-Cox if applying, or handle error
                                                if (
                                                    (
                                                        df_test_processed[
                                                            col_to_transform
                                                        ]
                                                        <= 0
                                                    )
                                                    .any()
                                                    .any()
                                                ):
                                                    st.warning(
                                                        f"Warning: Test data for '{col_to_transform}' contains non-positive values. Box-Cox with training lambda might fail or produce NaNs for these values."
                                                    )
                                                    # Attempt transformation, NaNs might result for non-positive values
                                                    df_test_processed[
                                                        col_to_transform
                                                    ] = stats.boxcox(
                                                        df_test_processed[
                                                            col_to_transform
                                                        ]
                                                        + 1e-6,
                                                        lmbda=lambda_value,
                                                    )  # Adding small constant if strictly positive needed, or handle differently
                                                else:
                                                    df_test_processed[
                                                        col_to_transform
                                                    ] = stats.boxcox(
                                                        df_test_processed[
                                                            col_to_transform
                                                        ],
                                                        lmbda=lambda_value,
                                                    )

                                            if (
                                                st.session_state["validation_df"]
                                                is not None
                                            ):
                                                if (
                                                    col_to_transform
                                                    in df_val_processed.columns
                                                ):
                                                    # Ensure test data is also positive for Box-Cox if applying, or handle error
                                                    if (
                                                        (
                                                            df_val_processed[
                                                                col_to_transform
                                                            ]
                                                            <= 0
                                                        )
                                                        .any()
                                                        .any()
                                                    ):
                                                        st.warning(
                                                            f"Warning: Validation data for '{col_to_transform}' contains non-positive values. Box-Cox with training lambda might fail or produce NaNs for these values."
                                                        )
                                                        # Attempt transformation, NaNs might result for non-positive values
                                                        df_val_processed[
                                                            col_to_transform
                                                        ] = stats.boxcox(
                                                            df_val_processed[
                                                                col_to_transform
                                                            ]
                                                            + 1e-6,
                                                            lmbda=lambda_value,
                                                        )  # Adding small constant if strictly positive needed, or handle differently
                                                    else:
                                                        df_val_processed[
                                                            col_to_transform
                                                        ] = stats.boxcox(
                                                            df_val_processed[
                                                                col_to_transform
                                                            ],
                                                            lmbda=lambda_value,
                                                        )

                                            message_text = f":material/check_circle: Box-Cox transform applied to '{col_to_transform}' successfully!"
                                        except Exception as e:
                                            message_text = f":material/error: Error during Box-Cox on '{col_to_transform}': {e}"
                                            original_message_type = "error"

                                case "Yeo-Johnson Transform":
                                    try:
                                        transformed_train_data, lambda_value = (
                                            stats.yeojohnson(
                                                df_train_processed[col_to_transform]
                                            )
                                        )
                                        df_train_processed[col_to_transform] = (
                                            transformed_train_data
                                        )

                                        if (
                                            col_to_transform
                                            in df_test_processed.columns
                                        ):
                                            df_test_processed[col_to_transform] = (
                                                stats.yeojohnson(
                                                    df_test_processed[col_to_transform],
                                                    lmbda=lambda_value,
                                                )
                                            )

                                        if (
                                            st.session_state["validation_df"]
                                            is not None
                                        ):
                                            df_val_processed[col_to_transform] = (
                                                stats.yeojohnson(
                                                    df_val_processed[col_to_transform],
                                                    lmbda=lambda_value,
                                                )
                                            )

                                        message_text = f":material/check_circle: Yeo-Johnson transform applied to '{col_to_transform}' successfully!"
                                    except Exception as e:
                                        message_text = f":material/error: Error during Yeo-Johnson on '{col_to_transform}': {e}"
                                        original_message_type = "error"

                            # Update session state with copies of the *processed* DataFrames.
                            st.session_state["current_df"] = df_train_processed.copy()
                            st.session_state["test_df"] = df_test_processed.copy()

                            # OPTIONAL
                            if (
                                "train_df" in st.session_state
                                and st.session_state["train_df"] is not None
                            ):
                                st.session_state["train_df"] = df_train_processed.copy()

                            if st.session_state["validation_df"] is not None:
                                st.session_state["validation_df"] = (
                                    df_val_processed.copy()
                                )

                            if original_message_type == "success":
                                st.session_state.is_data_transformed = True

                            st.session_state["message"] = (
                                original_message_type,
                                message_text,
                            )

                            if st.session_state["validation_df"] is not None:
                                return (
                                    df_train_processed,
                                    df_val_processed,
                                    df_test_processed,
                                )

                            return df_train_processed, df_test_processed

                        else:  # No split
                            df_processed = df_input.copy()
                            col_to_transform = params["col_to_transform"]
                            transform_method = params["transform_method"]

                            match transform_method:
                                case "Log Transform":
                                    if (df_processed[col_to_transform] < 0).any():
                                        message_type = "warning"
                                        message_text = f":material/warning: Log transform (log1p) skipped for column '{col_to_transform}' due to negative values."
                                    else:
                                        # Apply log(1+x) for stability around zero
                                        df_processed[col_to_transform] = np.log1p(
                                            df_processed[col_to_transform]
                                        )
                                        message_type = "success"
                                        message_text = f":material/check_circle: Log transform (log1p) applied successfully!"

                                case "Box-Cox Transform":
                                    if (df_processed[col_to_transform] <= 0).any():
                                        message_type = "warning"
                                        message_text = f':material/warning: Box-Cox transform skipped for column "{col_to_transform}" as it contains non-positive values.'
                                    else:
                                        # stats.boxcox returns transformed data and the optimal lambda
                                        transformed_data, _ = stats.boxcox(
                                            df_processed[col_to_transform]
                                        )
                                        df_processed[col_to_transform] = (
                                            transformed_data
                                        )
                                        df_processed[col_to_transform] = np.log1p(
                                            df_processed[col_to_transform]
                                        )
                                        message_type = "success"
                                        message_text = f":material/check_circle: Data transform (box-cos) applied successfully!"

                                case "Yeo-Johnson Transform":
                                    # Yeo-Johnson works with positive, zero, and negative values
                                    # stats.yeojohnson returns transformed data and the optimal lambda
                                    transformed_data, _ = stats.yeojohnson(
                                        df_processed[col_to_transform]
                                    )
                                    df_processed[col_to_transform] = transformed_data
                                    message_type = "success"
                                    message_text = f":material/check_circle: Data transform (Yeo-Johnson) applied successfully!"

                            # Update session_state
                            st.session_state["current_df"] = df_processed

                            # Set transformed data to True
                            if message_text == "success":
                                st.session_state.is_data_transformed = True

                            # Send message
                            st.session_state["message"] = (message_type, message_text)

                            return df_processed

                    case "variance_threshold":
                        if is_split:
                            # Get copies of the current train and test data from session state.
                            df_train_input = st.session_state["current_df"].copy()
                            df_test_input = st.session_state["test_df"].copy()

                            df_train_processed = df_train_input.copy()
                            df_test_processed = df_test_input.copy()

                            if st.session_state["validation_df"] is not None:
                                df_val_input = st.session_state["validation_df"].copy()
                                df_val_processed = df_val_input.copy()

                            threshold = float(params["threshold"])

                            # Identify numeric and non-numeric columns from the input dataframe(s)
                            num_cols_train = df_train_input.select_dtypes(
                                include=np.number
                            ).columns.tolist()
                            non_num_cols_train = df_train_input.select_dtypes(
                                exclude=np.number
                            ).columns.tolist()

                            # For test, align numeric columns with train for consistency if possible, or use its own
                            num_cols_test = [
                                col
                                for col in num_cols_train
                                if col in df_test_input.columns
                                and pd.api.types.is_numeric_dtype(df_test_input[col])
                            ]
                            non_num_cols_test = df_test_input.select_dtypes(
                                exclude=np.number
                            ).columns.tolist()

                            if st.session_state["validation_df"] is not None:
                                num_cols_val = [
                                    col
                                    for col in num_cols_train
                                    if col in df_val_input.columns
                                    and pd.api.types.is_numeric_dtype(df_val_input[col])
                                ]
                                non_num_cols_val = df_val_input.select_dtypes(
                                    exclude=np.number
                                ).columns.tolist()

                                removed_features_val = []

                            selector = VarianceThreshold(threshold=threshold)
                            removed_features_train = []
                            removed_features_test = []

                            if not num_cols_train:
                                message_text = ":material/warning: No numeric features in training data to apply variance threshold."
                                st.session_state["message"] = ("warning", message_text)
                                st.session_state["current_df"] = df_train_input.copy()
                                st.session_state["test_df"] = df_test_input.copy()
                                if (
                                    "train_df" in st.session_state
                                    and st.session_state["train_df"] is not None
                                ):
                                    st.session_state["train_df"] = df_train_input.copy()

                                if st.session_state["validation_df"] is not None:
                                    st.session_state["validation_df"] = (
                                        df_val_input.copy()
                                    )
                                    return df_train_input, df_val_input, df_test_input

                                return df_train_input, df_test_input

                            # Fit on training numeric data
                            selector.fit(df_train_input[num_cols_train])
                            selected_mask_train = selector.get_support()
                            selected_numeric_features_train = (
                                df_train_input[num_cols_train]
                                .columns[selected_mask_train]
                                .tolist()
                            )
                            removed_features_train = list(
                                set(num_cols_train)
                                - set(selected_numeric_features_train)
                            )

                            # Reconstruct train DataFrame
                            df_train_processed = pd.concat(
                                [
                                    df_train_input[non_num_cols_train],
                                    df_train_input[selected_numeric_features_train],
                                ],
                                axis=1,
                            ).reindex(
                                columns=non_num_cols_train
                                + selected_numeric_features_train
                            )

                            # Transform test numeric data (if it has numeric columns)
                            if num_cols_test:
                                # Ensure selector is applied only to columns it was fit on that also exist in test
                                common_num_cols_for_test_transform = [
                                    col
                                    for col in num_cols_train
                                    if col in df_test_input[num_cols_test].columns
                                ]
                                if common_num_cols_for_test_transform:
                                    # Create a sub-selector or filter data for transform
                                    test_numeric_transformed_data = selector.transform(
                                        df_test_input[num_cols_train]
                                    )  # Use num_cols_train for consistent feature set
                                    selected_numeric_features_test = (
                                        df_test_input[num_cols_train]
                                        .columns[selected_mask_train]
                                        .tolist()
                                    )  # Same mask as train

                                    df_test_processed_numeric = pd.DataFrame(
                                        test_numeric_transformed_data,
                                        columns=selected_numeric_features_test,
                                        index=df_test_input.index,
                                    )

                                    # Reconstruct test DataFrame
                                    df_test_processed = pd.concat(
                                        [
                                            df_test_input[
                                                non_num_cols_test
                                            ].reset_index(drop=True),
                                            df_test_processed_numeric.reset_index(
                                                drop=True
                                            ),
                                        ],
                                        axis=1,
                                    )
                                    # Ensure original column order for non-numeric part, then selected numeric
                                    final_test_cols = [
                                        col
                                        for col in non_num_cols_test
                                        if col in df_test_processed.columns
                                    ] + [
                                        col
                                        for col in selected_numeric_features_test
                                        if col in df_test_processed.columns
                                    ]
                                    df_test_processed = df_test_processed[
                                        final_test_cols
                                    ]

                                else:
                                    message_text = "Test set numeric columns did not align sufficiently with training set for VarianceThreshold transformation."
                                    message_type = "warning"

                            else:  # No numeric columns in test set
                                df_test_processed = (
                                    df_test_input.copy()
                                )  # No change to test if no numeric cols

                            if st.session_state["validation_df"] is not None:
                                # Transform validation numeric data (if it has numeric columns)
                                if num_cols_val:
                                    # Ensure selector is applied only to columns it was fit on that also exist in validation
                                    common_num_cols_for_val_transform = [
                                        col
                                        for col in num_cols_train
                                        if col in df_val_input[num_cols_val].columns
                                    ]
                                    if common_num_cols_for_val_transform:
                                        # Create a sub-selector or filter data for transform
                                        val_numeric_transformed_data = selector.transform(
                                            df_val_input[num_cols_train]
                                        )  # Use num_cols_train for consistent feature set
                                        selected_numeric_features_val = (
                                            df_val_input[num_cols_train]
                                            .columns[selected_mask_train]
                                            .tolist()
                                        )  # Same mask as train

                                        df_val_processed_numeric = pd.DataFrame(
                                            val_numeric_transformed_data,
                                            columns=selected_numeric_features_val,
                                            index=df_val_input.index,
                                        )

                                        # Reconstruct validation DataFrame
                                        df_val_processed = pd.concat(
                                            [
                                                df_val_input[
                                                    non_num_cols_val
                                                ].reset_index(drop=True),
                                                df_val_processed_numeric.reset_index(
                                                    drop=True
                                                ),
                                            ],
                                            axis=1,
                                        )
                                        # Ensure original column order for non-numeric part, then selected numeric
                                        final_val_cols = [
                                            col
                                            for col in non_num_cols_val
                                            if col in df_val_processed.columns
                                        ] + [
                                            col
                                            for col in selected_numeric_features_val
                                            if col in df_val_processed.columns
                                        ]
                                        df_val_processed = df_val_processed[
                                            final_val_cols
                                        ]

                                    else:
                                        message_type = "warning"
                                        message_text = "Validation set numeric columns did not align sufficiently with training set for VarianceThreshold transformation."

                                else:  # No numeric columns in validation set
                                    df_val_processed = (
                                        df_val_input.copy()
                                    )  # No change to validation if no numeric cols

                            if removed_features_train:
                                message_text_train = (
                                    f"Removed: {removed_features_train}"
                                )
                                message_type = "success"
                            else:
                                message_text_train = "No features removed."
                                message_type = "info"

                            message_text = f":material/check_circle: VarianceThreshold applied. {message_text_train}"

                            # Update session_state
                            st.session_state["current_df"] = df_train_processed.copy()
                            st.session_state["test_df"] = df_test_processed.copy()
                            # OPTIONAL
                            if (
                                "train_df" in st.session_state
                                and st.session_state["train_df"] is not None
                            ):
                                st.session_state["train_df"] = df_train_processed.copy()

                            # Send message
                            st.session_state["message"] = (message_type, message_text)

                            if st.session_state["validation_df"] is not None:
                                st.session_state["validation_df"] = (
                                    df_val_processed.copy()
                                )
                                return (
                                    df_train_processed,
                                    df_val_processed,
                                    df_test_processed,
                                )

                            return df_train_processed, df_test_processed

                        else:  # No split
                            data = df_input.copy()
                            df_processed = df_input.copy()
                            features = data[params["num_cols"]]
                            threshold = float(params["threshold"])
                            selector = VarianceThreshold(threshold=threshold)

                            # Fit the selector to the numeric features
                            selector.fit(features)

                            # Get the boolean mask of selected features
                            selected_mask = selector.get_support()

                            # Get the names of the selected numeric features
                            selected_numeric_features = features.columns[selected_mask]

                            # Reconstruct DataFrame
                            # Keep non-numeric columns
                            non_numeric_cols = params["non_num_cols"]
                            selected_all_features = list(non_numeric_cols) + list(
                                selected_numeric_features
                            )

                            # Return the DataFrame with selected features (maintaining original order where possible)
                            df_processed = data[selected_all_features]

                            removed_features = list(
                                set(params["num_cols"]) - set(selected_numeric_features)
                            )
                            if len(removed_features) <= 0:
                                message_text = ":material/check_circle: VarianceThreshold applied. No features removed."
                                message_type = "info"
                            else:
                                message_text = f":material/check_circle: VarianceThreshold applied. Removed: {removed_features}"
                                message_type = "success"

                            # Update session_state
                            st.session_state["current_df"] = df_processed.copy()
                            # Send message
                            st.session_state["message"] = (message_type, message_text)

                            return df_processed

                    case "pca":
                        is_split = st.session_state.get("is_data_split", False)
                        n_components = params.get("n_components")

                        apply_scaling_before_pca = not params.get(
                            "is_data_scaled",
                            st.session_state.get("is_data_scaled", False),
                        )
                        scaler_instance = None

                        if is_split:
                            # Get copies of the current train and test data from session state.
                            df_train_input = st.session_state["current_df"].copy()
                            df_test_input = st.session_state["test_df"].copy()

                            # Train cols
                            numeric_cols_train = df_train_input.select_dtypes(
                                include=np.number
                            ).columns.tolist()
                            non_numeric_cols_train = df_train_input.select_dtypes(
                                exclude=np.number
                            ).columns.tolist()
                            # Test cols
                            numeric_cols_test = df_test_input.select_dtypes(
                                include=np.number
                            ).columns.tolist()
                            non_numeric_cols_test = df_test_input.select_dtypes(
                                exclude=np.number
                            ).columns.tolist()

                            if st.session_state["validation_df"] is not None:
                                df_val_input = st.session_state["validation_df"].copy()
                                # Validation cols
                                numeric_cols_val = df_val_input.select_dtypes(
                                    include=np.number
                                ).columns.tolist()
                                non_numeric_cols_val = df_val_input.select_dtypes(
                                    exclude=np.number
                                ).columns.tolist()

                            if not numeric_cols_train:
                                st.session_state["message"] = (
                                    "warning",
                                    ":material/warning: PCA requires numeric data. No numeric columns found in training set.",
                                )
                                st.session_state["current_df"] = df_train_input.copy()
                                st.session_state["test_df"] = df_test_input.copy()

                                if (
                                    "train_df" in st.session_state
                                    and st.session_state["train_df"] is not None
                                ):
                                    st.session_state["train_df"] = df_train_input.copy()

                                if st.session_state["validation_df"] is not None:
                                    st.session_state["validation_df"] = (
                                        df_val_input.copy()
                                    )
                                    return df_train_input, df_val_input, df_test_input

                                return df_train_input, df_test_input

                            x_train = df_train_input[numeric_cols_train].copy()
                            x_test = df_test_input[numeric_cols_test].copy()
                            if st.session_state["validation_df"] is not None:
                                x_val = df_val_input[numeric_cols_val].copy()

                            # Scaling
                            if apply_scaling_before_pca:
                                selected_scaler_method = params.get("selected_scaler")
                                if selected_scaler_method == "MinMaxScaler (0-1 range)":
                                    scaler_instance = MinMaxScaler()
                                elif selected_scaler_method == "Robust Scaler":
                                    scaler_instance = RobustScaler()
                                else:
                                    scaler_instance = StandardScaler()
                                    st.info(
                                        "PCA: No specific scaler chosen or data not pre-scaled. Applying StandardScaler.",
                                        icon="ℹ️",
                                    )

                                x_train_scaled = scaler_instance.fit_transform(x_train)
                                x_train = pd.DataFrame(
                                    x_train_scaled,
                                    columns=x_train.columns,
                                    index=x_train.index,
                                )

                                if not x_test.empty and all(
                                    col in x_test.columns for col in x_train.columns
                                ):
                                    x_test_scaled = scaler_instance.transform(
                                        x_test[x_train.columns]
                                    )
                                    x_test = pd.DataFrame(
                                        x_test_scaled,
                                        columns=x_train.columns,
                                        index=x_test.index,
                                    )

                                    if st.session_state["validation_df"] is not None:
                                        x_val_scaled = scaler_instance.transform(
                                            x_val[x_train.columns]
                                        )
                                        x_val = pd.DataFrame(
                                            x_val_scaled,
                                            columns=x_train.columns,
                                            index=x_val.index,
                                        )
                                else:
                                    st.session_state["message"] = (
                                        "warning",
                                        "PCA: Test set columns do not align for scaling. Attempting fallback...",
                                    )
                                    common_cols = x_train.columns.intersection(
                                        x_test.columns
                                    )
                                    if not common_cols.empty:
                                        x_test_scaled_common = (
                                            scaler_instance.transform(
                                                x_test[common_cols]
                                            )
                                        )
                                        x_test = pd.DataFrame(
                                            x_test_scaled_common,
                                            columns=common_cols,
                                            index=x_test.index,
                                        )
                                    else:
                                        x_test = pd.DataFrame(index=x_test.index)

                                st.session_state.is_data_scaled = True

                            # PCA
                            max_pca_components = min(x_train.shape[0], x_train.shape[1])
                            if (
                                n_components is None
                                or n_components > max_pca_components
                            ):
                                n_components = max_pca_components

                            pca_transformer = PCA(n_components=n_components)
                            pc_columns = [f"PC{i+1}" for i in range(n_components)]

                            principal_components_train = pca_transformer.fit_transform(
                                x_train
                            )
                            df_train_pca = pd.DataFrame(
                                principal_components_train,
                                columns=pc_columns,
                                index=df_train_input.index,
                            )
                            df_train_processed = pd.concat(
                                [
                                    df_train_input[non_numeric_cols_train].reset_index(
                                        drop=True
                                    ),
                                    df_train_pca.reset_index(drop=True),
                                ],
                                axis=1,
                            )

                            if (
                                not x_test.empty
                                and x_test.shape[1] == x_train.shape[1]
                                and all(x_test.columns == x_train.columns)
                            ):
                                principal_components_test = pca_transformer.transform(
                                    x_test
                                )
                                df_test_pca = pd.DataFrame(
                                    principal_components_test,
                                    columns=pc_columns,
                                    index=df_test_input.index,
                                )
                                df_test_processed = pd.concat(
                                    [
                                        df_test_input[
                                            non_numeric_cols_test
                                        ].reset_index(drop=True),
                                        df_test_pca.reset_index(drop=True),
                                    ],
                                    axis=1,
                                )

                                if st.session_state["validation_df"] is not None:
                                    principal_components_val = (
                                        pca_transformer.transform(x_val)
                                    )
                                    df_val_pca = pd.DataFrame(
                                        principal_components_val,
                                        columns=pc_columns,
                                        index=df_val_input.index,
                                    )
                                    df_val_processed = pd.concat(
                                        [
                                            df_val_input[
                                                non_numeric_cols_val
                                            ].reset_index(drop=True),
                                            df_val_pca.reset_index(drop=True),
                                        ],
                                        axis=1,
                                    )

                            else:
                                st.session_state["message"] = (
                                    "warning",
                                    "PCA: Test set could not be transformed. Retaining only non-numeric columns.",
                                )
                                df_test_processed = (
                                    df_test_input[non_numeric_cols_test].copy()
                                    if non_numeric_cols_test
                                    else pd.DataFrame(index=df_test_input.index)
                                )

                                if st.session_state["validation_df"] is not None:
                                    df_val_processed = (
                                        df_val_input[non_numeric_cols_val].copy()
                                        if non_numeric_cols_val
                                        else pd.DataFrame(index=df_val_input.index)
                                    )

                            # Set flag to True
                            st.session_state.pca_done = True

                            # Send message
                            message_type = "success"
                            message_text = f":material/check_circle: PCA applied. Reduced to {n_components} components."
                            st.session_state["message"] = (message_type, message_text)

                            # Update session state
                            st.session_state["current_df"] = df_train_processed.copy()
                            st.session_state["test_df"] = df_test_processed.copy()
                            # OPTIONAL
                            if (
                                "train_df" in st.session_state
                                and st.session_state["train_df"] is not None
                            ):
                                st.session_state["train_df"] = df_train_processed.copy()

                            # Explained Variance (shared)
                            st.session_state["explained_variance"] = []
                            explained_variance_ratio = (
                                pca_transformer.explained_variance_ratio_
                            )
                            cumulative_variance_ratio = np.cumsum(
                                explained_variance_ratio
                            )
                            for i, ratio in enumerate(explained_variance_ratio):
                                st.session_state["explained_variance"].append(
                                    f"Principal Component {i+1}: {ratio:.4f} (Cumulative: {cumulative_variance_ratio[i]:.4f})"
                                )

                            st.session_state["message"] = (
                                "success",
                                f":material/check_circle: PCA applied. Reduced to {n_components} components.",
                            )

                            if st.session_state["validation_df"] is not None:
                                st.session_state["validation_df"] = (
                                    df_val_processed.copy()
                                )
                                return (
                                    df_train_processed,
                                    df_val_processed,
                                    df_test_processed,
                                )

                            return df_train_processed, df_test_processed

                        else:  # No split
                            df_input = st.session_state["current_df"].copy()
                            numeric_cols = df_input.select_dtypes(
                                include=np.number
                            ).columns.tolist()
                            non_numeric_cols = df_input.select_dtypes(
                                exclude=np.number
                            ).columns.tolist()

                            if not numeric_cols:
                                st.session_state["message"] = (
                                    "warning",
                                    ":material/warning: PCA requires numeric data. No numeric columns found.",
                                )
                                st.session_state["current_df"] = df_input.copy()
                                return df_input

                            x_data = df_input[numeric_cols].copy()

                            # Scaling
                            if apply_scaling_before_pca:
                                selected_scaler_method = params.get("selected_scaler")
                                if selected_scaler_method == "MinMaxScaler (0-1 range)":
                                    scaler_instance = MinMaxScaler()
                                elif selected_scaler_method == "Robust Scaler":
                                    scaler_instance = RobustScaler()
                                else:
                                    scaler_instance = StandardScaler()
                                    st.info(
                                        "PCA: No specific scaler chosen or data not pre-scaled. Applying StandardScaler.",
                                        icon="ℹ️",
                                    )

                                x_data_scaled = scaler_instance.fit_transform(x_data)
                                x_data = pd.DataFrame(
                                    x_data_scaled,
                                    columns=x_data.columns,
                                    index=x_data.index,
                                )

                                st.session_state.is_data_scaled = True
                                st.info(
                                    f"Data scaled with {scaler_instance.__class__.__name__} before PCA.",
                                    icon="ℹ️",
                                )

                            # PCA
                            max_pca_components = min(x_data.shape[0], x_data.shape[1])
                            if (
                                n_components is None
                                or n_components > max_pca_components
                            ):
                                n_components = max_pca_components
                                st.info(
                                    f"PCA: n_components adjusted to {n_components} (max available features/samples).",
                                    icon="ℹ️",
                                )

                            pca_transformer = PCA(n_components=n_components)
                            pc_columns = [f"PC{i+1}" for i in range(n_components)]
                            principal_components_data = pca_transformer.fit_transform(
                                x_data
                            )
                            df_data_pca = pd.DataFrame(
                                principal_components_data,
                                columns=pc_columns,
                                index=df_input.index,
                            )
                            df_processed = pd.concat(
                                [
                                    df_input[non_numeric_cols].reset_index(drop=True),
                                    df_data_pca.reset_index(drop=True),
                                ],
                                axis=1,
                            )

                            # Explained Variance (shared)
                            st.session_state["explained_variance"] = []
                            explained_variance_ratio = (
                                pca_transformer.explained_variance_ratio_
                            )
                            cumulative_variance_ratio = np.cumsum(
                                explained_variance_ratio
                            )
                            for i, ratio in enumerate(explained_variance_ratio):
                                st.session_state["explained_variance"].append(
                                    f"Principal Component {i+1}: {ratio:.4f} (Cumulative: {cumulative_variance_ratio[i]:.4f})"
                                )

                            # Update session state
                            st.session_state["current_df"] = df_processed.copy()

                            # Set flag to True
                            st.session_state.pca_done = True

                            # Send message
                            message_type = "success"
                            message_text = f":material/check_circle: PCA applied. Reduced to {n_components} components."
                            st.session_state["message"] = (message_type, message_text)

                            return df_processed

                    case "feature_selection":
                        col = params["target_col"]

                        # Separate features and target
                        X = df_out.drop(columns=[col])
                        y = df_out[col]

                        # Handle non-numeric features
                        X = pd.get_dummies(X, drop_first=True)

                        # Encode categorical target if needed
                        if y.dtype == "object" or y.dtype.name == "category":
                            y = LabelEncoder().fit_transform(y)

                        selector = SelectKBest(score_func=f_classif, k=params["k"])
                        X_new = selector.fit_transform(X, y)
                        selected_features = X.columns[selector.get_support()]
                        df_out = pd.concat(
                            [df_out[selected_features], df_out[params["target_col"]]],
                            axis=1,
                        )
                        # st.success(f"Selected Top {k} Features: {selected_features.tolist()}")

                    case _:
                        message_text = (
                            f":material/warning: Unknown operation: {op_type}"
                        )
                        message_type = "warning"
                        st.session_state["message"] = (message_type, message_text)

            except Exception as e:
                message_text = f":material/error: Error applying {op_type} (ID: {op_spec.get('id', 'N/A')}): {e}. Returning previous state."
                message_type = "error"
                st.session_state["message"] = (message_type, message_text)
                return df

        def compute_current_df():
            """
            Recomputes the current DataFrame state by applying all history operations
            sequentially starting from the initial DataFrame.
            """
            df_train_computed = st.session_state["initial_df"].copy()
            df_test_computed = pd.DataFrame()
            df_val_computed = pd.DataFrame()

            for op_spec in st.session_state["operations_history"]:

                result = apply_operation(df_train_computed.copy(), op_spec)
                if isinstance(result, tuple) and len(result) == 2:
                    df_train_computed, df_test_computed = result
                elif isinstance(result, tuple) and len(result) == 3:
                    df_train_computed, df_val_computed, df_test_computed = result
                else:
                    df_train_computed = result

            st.session_state["current_df"] = df_train_computed
            if not df_test_computed.empty:
                st.session_state["test_df"] = df_test_computed
            if not df_val_computed.empty:
                st.session_state["validation_df"] = df_val_computed

        def add_operation(op_type, params):
            """
            Adds a new operation to the history, recomputes the current state,
            and triggers a UI refresh.
            """
            op_spec = {
                "id": uuid.uuid4(),
                "op_type": op_type,
                "params": deepcopy(params),  # Use deepcopy for parameter safety
            }
            st.session_state["operations_history"].append(op_spec)
            compute_current_df()
            message = st.session_state["message"]
            match message[0]:
                case "success":
                    st.success(message[1])
                case "warning":
                    st.warning(message[1])
                    # Remove recently added operation if the message type was "warning"
                    st.session_state["operations_history"].pop()
                    compute_current_df()
                case "info":
                    st.info(message[1])
                    # Remove recently added operation if the message type was "info"
                    st.session_state["operations_history"].pop()
                    compute_current_df()
                case "error":
                    st.error(message[1])
                    # Remove recently added operation if the message type was "info"
                    st.session_state["operations_history"].pop()
                    compute_current_df()

            time.sleep(2)
            st.rerun()

        # --- Button to Undo Last ---
        st.sidebar.subheader("Undo Last Operation")
        if st.sidebar.button(
            "Undo", type="primary", icon=":material/undo:", use_container_width=True
        ):
            if st.session_state["operations_history"]:

                if st.session_state["operations_history"][-1]["op_type"] == "pca":
                    st.session_state.is_data_scaled = False
                    st.session_state.pca_done = False
                    st.session_state.verify_clicked = False
                    # st.session_state.is_data_encoded = False
                    st.session_state["explained_variance"] = []

                    # # Remove associated 'encode_columns' operation if it exists
                    # for i, op in enumerate(st.session_state["operations_history"]):
                    #     if op['op_type'] == 'encode_columns':
                    #         del st.session_state["operations_history"][i]
                    #         break  # Remove only the first occurrence

                elif (
                    st.session_state["operations_history"][-1]["op_type"]
                    == "scale_columns"
                ):
                    st.session_state.is_data_scaled = False

                elif (
                    st.session_state["operations_history"][-1]["op_type"]
                    == "encode_columns"
                ):
                    st.session_state.is_data_encoded = False

                elif (
                    st.session_state["operations_history"][-1]["op_type"]
                    == "transform_columns"
                ):
                    st.session_state.is_data_transformed = False

                elif (
                    st.session_state["operations_history"][-1]["op_type"]
                    == "train_test_split"
                ):
                    st.session_state["train_df"] = None
                    st.session_state["test_df"] = None
                    st.session_state["initial_train_df"] = None
                    st.session_state["initial_test_df"] = None
                    st.session_state.is_data_split = False

                elif (
                    st.session_state["operations_history"][-1]["op_type"]
                    == "train_test_val_split"
                ):
                    st.session_state["train_df"] = None
                    st.session_state["test_df"] = None
                    st.session_state["validation_df"] = None
                    st.session_state["initial_train_df"] = None
                    st.session_state["initial_test_df"] = None
                    st.session_state["initial_validation_df"] = None
                    st.session_state.is_data_split = False

                st.session_state[
                    "operations_history"
                ].pop()  # Remove last entry of the list
                compute_current_df()

            st.rerun()

        # --- Button to Reset All ---
        st.sidebar.subheader("Reset All")

        if st.sidebar.button(
            "Reset", type="primary", icon=":material/undo:", use_container_width=True
        ):
            st.session_state["operations_history"] = []  # Clear history
            st.session_state["explained_variance"] = []
            st.session_state["train_df"] = None
            st.session_state["test_df"] = None
            st.session_state["validation_df"] = None
            st.session_state["initial_train_df"] = None
            st.session_state["initial_test_df"] = None
            st.session_state["initial_validation_df"] = None
            st.session_state.pca_done = False
            st.session_state.verify_clicked = False
            st.session_state.is_data_scaled = False
            st.session_state.is_data_encoded = False
            st.session_state.is_data_transformed = False
            st.session_state.is_data_split = False
            compute_current_df()
            st.rerun()

        # --- UI tabs for Operations ---
        tab_labels = [
            "Exploration",
            "Cleaning",
            # Missing Values, Outliers, Duplicated, Drop col, Data type conver
            "EDA",
            "Splitting",
            "Transformation",
            # Scaling, Encoding cat, Power trans,
            "Reduction",  # Feature selection
            # PCA, Variance threshold
            "Feature Engineering",
            "Binarization",
            "Export",
            "Operations History",
        ]

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
            tab_labels
        )

        with tab1:  # --- Data Ingestion/Exploration ---
            st.subheader("🔍 Dataset Display")

            if (
                st.session_state.is_data_split == False
            ):  # --- DataFrame Display (No Split) ---
                with st.container(border=True):
                    df_display_Type = st.radio(
                        "Select the way you would like to display the DataFrame",
                        ["DataFrame Head", "DataFrame Tail", "Full DataFrame"],
                        horizontal=True,
                    )

                    if df_display_Type == "DataFrame Head":
                        st.dataframe(
                            st.session_state["current_df"].head(),
                            use_container_width=True,
                        )
                    elif df_display_Type == "DataFrame Tail":
                        st.dataframe(
                            st.session_state["current_df"].tail(),
                            use_container_width=True,
                        )
                    elif df_display_Type == "Full DataFrame":
                        st.dataframe(
                            st.session_state["current_df"], use_container_width=True
                        )

                    # --- Display DataFrame shape ---
                    initial_rows = st.session_state["initial_df"].shape[0]
                    initial_cols = st.session_state["initial_df"].shape[1]
                    current_rows = st.session_state["current_df"].shape[0]
                    current_cols = st.session_state["current_df"].shape[1]

                    # Calculate deltas
                    delta_rows = current_rows - initial_rows
                    delta_cols = current_cols - initial_cols

                    # Helper function for arrow formatting
                    def format_delta(delta):
                        if delta > 0:
                            return f"<span style='color:rgba(45, 186, 89, 0.8)'>▲ {delta}</span>"
                        elif delta < 0:
                            return f"<span style='color:rgba(240, 48, 48, 0.8)'>▼ {abs(delta)}</span>"
                        else:
                            return f"<span style='color:gray'>➖ 0</span>"

                    col_1, col_2 = st.columns(2)

                    with col_1:  # --- Initial shape ---
                        with st.container(border=True):
                            col1, col2, col3 = st.columns([2, 2, 2])

                            with col1:
                                st.markdown("###### Initial DF")
                            with col2:
                                st.markdown(f"Rows: **{initial_rows}**")
                            with col3:
                                st.markdown(f"Columns: **{initial_cols}**")

                    with col_2:  # --- Current Shape ---
                        with st.container(border=True):
                            col1, col2, col3 = st.columns([2, 2, 2])

                            with col1:
                                st.markdown("###### Current DF")
                            with col2:
                                st.markdown(
                                    f"Rows: **{current_rows}**  {format_delta(delta_rows)}",
                                    unsafe_allow_html=True,
                                )
                            with col3:
                                st.markdown(
                                    f"Columns: **{current_cols}**  {format_delta(delta_cols)}",
                                    unsafe_allow_html=True,
                                )

                    if st.session_state["explained_variance"]:
                        with st.container():
                            st.markdown("##### Explained Variance:")
                            code_output = "\n".join(
                                st.session_state["explained_variance"]
                            )
                            st.code(code_output, language="python")

                        # Alternative method:
                        # for line in st.session_state['explained_variance']:
                        #     st.markdown(line)

            elif st.session_state.is_data_split:
                with st.container(border=True):  # --- Display Train df ---
                    st.markdown("#### Train set")
                    df_display_Type = st.radio(
                        "Select the way you would like to display the DataFrame",
                        ["DataFrame Head", "DataFrame Tail", "Full DataFrame"],
                        horizontal=True,
                        key="train_display",
                    )

                    if df_display_Type == "DataFrame Head":
                        st.dataframe(
                            st.session_state["current_df"].head(),
                            use_container_width=True,
                        )
                    elif df_display_Type == "DataFrame Tail":
                        st.dataframe(
                            st.session_state["current_df"].tail(),
                            use_container_width=True,
                        )
                    elif df_display_Type == "Full DataFrame":
                        st.dataframe(
                            st.session_state["current_df"], use_container_width=True
                        )

                    # Initial Train
                    initial_train_rows = st.session_state["initial_train_df"].shape[0]
                    initial_train_cols = st.session_state["initial_train_df"].shape[1]

                    # Current Train
                    train_rows = st.session_state["current_df"].shape[0]
                    train_cols = st.session_state["current_df"].shape[1]

                    # Calculate Train Deltas
                    delta_train_rows = train_rows - initial_train_rows
                    delta_train_cols = train_cols - initial_train_cols

                    # Helper function for arrow formatting
                    def format_delta(delta):
                        if delta > 0:
                            return f"<span style='color:rgba(45, 186, 89, 0.8)'>▲ {delta}</span>"
                        elif delta < 0:
                            return f"<span style='color:rgba(240, 48, 48, 0.8)'>▼ {abs(delta)}</span>"
                        else:
                            return f"<span style='color:gray'>➖ 0</span>"

                    col_1, col_2 = st.columns(2)

                    with col_1:  # --- Initial Train shape ---
                        with st.container(border=True):
                            col1, col2, col3 = st.columns([2, 2, 2])

                            with col1:
                                st.markdown("###### Initial Train")
                            with col2:
                                st.markdown(f"Rows: **{initial_train_rows}**")
                            with col3:
                                st.markdown(f"Columns: **{initial_train_cols}**")

                    with col_2:  # --- Current Train Shape ---
                        with st.container(border=True):
                            col1, col2, col3 = st.columns([2, 2, 2])

                            with col1:
                                st.markdown("###### Current Train")
                            with col2:
                                st.markdown(
                                    f"Rows: **{train_rows}** {format_delta(delta_train_rows)}",
                                    unsafe_allow_html=True,
                                )
                            with col3:
                                st.markdown(
                                    f"Columns: **{train_cols}** {format_delta(delta_train_cols)}",
                                    unsafe_allow_html=True,
                                )

                if st.session_state["validation_df"] is not None:
                    with st.container(border=True):  # --- Display Validation df ---
                        st.markdown("#### Validation set")
                        df_display_Type = st.radio(
                            "Select the way you would like to display the DataFrame",
                            ["DataFrame Head", "DataFrame Tail", "Full DataFrame"],
                            horizontal=True,
                            key="validation_display",
                        )

                        if df_display_Type == "DataFrame Head":
                            st.dataframe(
                                st.session_state["validation_df"].head(),
                                use_container_width=True,
                            )
                        elif df_display_Type == "DataFrame Tail":
                            st.dataframe(
                                st.session_state["validation_df"].tail(),
                                use_container_width=True,
                            )
                        elif df_display_Type == "Full DataFrame":
                            st.dataframe(
                                st.session_state["validation_df"],
                                use_container_width=True,
                            )

                        # Initial Validation
                        initial_val_rows = st.session_state[
                            "initial_validation_df"
                        ].shape[0]
                        initial_val_cols = st.session_state[
                            "initial_validation_df"
                        ].shape[1]

                        # Current Validation
                        val_rows = st.session_state["validation_df"].shape[0]
                        val_cols = st.session_state["validation_df"].shape[1]

                        # Calculate Validation Deltas
                        delta_val_rows = val_rows - initial_val_rows
                        delta_val_cols = val_cols - initial_val_cols

                        col_1, col_2 = st.columns(2)

                        with col_1:  # --- Initial Validation shape ---
                            with st.container(border=True):
                                col1, col2, col3 = st.columns([2, 2, 2])

                                with col1:
                                    st.markdown("###### Initial Validation")
                                with col2:
                                    st.markdown(f"Rows: **{initial_val_rows}**")
                                with col3:
                                    st.markdown(f"Columns: **{initial_val_cols}**")

                        with col_2:  # --- Current Validation Shape ---
                            with st.container(border=True):
                                col1, col2, col3 = st.columns([2, 2, 2])

                                with col1:
                                    st.markdown("###### Current Validation")
                                with col2:
                                    st.markdown(
                                        f"Rows: **{val_rows}** {format_delta(delta_val_rows)}",
                                        unsafe_allow_html=True,
                                    )
                                with col3:
                                    st.markdown(
                                        f"Columns: **{val_cols}** {format_delta(delta_val_cols)}",
                                        unsafe_allow_html=True,
                                    )

                with st.container(border=True):  # --- Display Test df ---
                    st.markdown("#### Test set")
                    df_display_Type = st.radio(
                        "Select the way you would like to display the DataFrame",
                        ["DataFrame Head", "DataFrame Tail", "Full DataFrame"],
                        horizontal=True,
                        key="test_display",
                    )

                    if df_display_Type == "DataFrame Head":
                        st.dataframe(
                            st.session_state["test_df"].head(), use_container_width=True
                        )
                    elif df_display_Type == "DataFrame Tail":
                        st.dataframe(
                            st.session_state["test_df"].tail(), use_container_width=True
                        )
                    elif df_display_Type == "Full DataFrame":
                        st.dataframe(
                            st.session_state["test_df"], use_container_width=True
                        )

                    # Initial Test
                    initial_test_rows = st.session_state["initial_test_df"].shape[0]
                    initial_test_cols = st.session_state["initial_test_df"].shape[1]

                    # Current Test
                    test_rows = st.session_state["test_df"].shape[0]
                    test_cols = st.session_state["test_df"].shape[1]

                    # Calculate Test Deltas
                    delta_test_rows = test_rows - initial_test_rows
                    delta_test_cols = test_cols - initial_test_cols

                    col_1, col_2 = st.columns(2)

                    with col_1:  # --- Initial Test shape ---
                        with st.container(border=True):
                            col1, col2, col3 = st.columns([2, 2, 2])

                            with col1:
                                st.markdown("###### Initial Test")
                            with col2:
                                st.markdown(f"Rows: **{initial_test_rows}**")
                            with col3:
                                st.markdown(f"Columns: **{initial_test_cols}**")

                    with col_2:  # --- Current Test Shape ---
                        with st.container(border=True):
                            col1, col2, col3 = st.columns([2, 2, 2])

                            with col1:
                                st.markdown("###### Current Test")
                            with col2:
                                st.markdown(
                                    f"Rows: **{test_rows}** {format_delta(delta_test_rows)}",
                                    unsafe_allow_html=True,
                                )
                            with col3:
                                st.markdown(
                                    f"Columns: **{test_cols}** {format_delta(delta_test_cols)}",
                                    unsafe_allow_html=True,
                                )

                if st.session_state[
                    "explained_variance"
                ]:  # --- Display Explained Variance ---
                    with st.container():
                        st.markdown("##### Explained Variance:")
                        code_output = "\n".join(st.session_state["explained_variance"])
                        st.code(code_output, language="python")

            with st.container(border=True):  # --- Rename Column ---
                st.markdown("#### Rename Column")
                available_columns_rename = st.session_state[
                    "current_df"
                ].columns.tolist()
                if not available_columns_rename:
                    st.warning("No columns available to rename.")
                else:
                    old_column = st.selectbox(
                        "Select column to rename",
                        available_columns_rename,
                        key="rename_old_select",
                    )
                    new_column = st.text_input(
                        "Enter new column name", key="rename_new_input"
                    )

                    if st.button("Apply", key="apply_rename_column"):
                        # Validation checks
                        if (
                            old_column
                            and new_column
                            and old_column != new_column
                            and new_column not in available_columns_rename
                        ):
                            params = {"old_name": old_column, "new_name": new_column}
                            add_operation("rename_column", params)
                        elif not old_column:
                            st.warning(
                                "Please select a column to rename.",
                                icon=":material/warning:",
                            )
                        elif not new_column:
                            st.warning(
                                "Please enter a new column name.",
                                icon=":material/warning:",
                            )
                        elif old_column == new_column:
                            st.warning(
                                "Old and new column names are the same.",
                                icon=":material/warning:",
                            )
                        elif new_column in available_columns_rename:
                            st.warning(
                                f"New column name **{new_column}** already exists.",
                                icon=":material/warning:",
                            )

        with tab2:  # --- Initial Data Cleaning ---
            st.subheader("🧹 Initial Data Cleaning")
            with st.container(border=True):  # --- Handling Missing Values ---
                st.markdown("####  Handle Missing Values")
                df = st.session_state["current_df"]
                cols_with_na = df.columns[df.isnull().any()].tolist()

                col1, col2, col3 = st.columns([1.2, 1.4, 2])

                if cols_with_na:
                    # if st.session_state["current_df"].isnull().sum().sum() > 0:
                    with col1:
                        fill_strategy = st.selectbox(
                            "Select a strategy",
                            [
                                "Drop",
                                "Fill with Mean",
                                "Fill with Median",
                                "Fill with Mode",
                                "Fill with Custom Value",
                            ],
                        )

                    with col2:
                        if fill_strategy == "Fill with Custom Value":
                            custom_value = st.text_input("Enter custom value:")

                            params = {
                                "fill_strategy": fill_strategy,
                                "cols_with_na": cols_with_na,
                                "custom_value": custom_value,
                            }

                        elif fill_strategy == "Drop":
                            axis = st.radio(
                                "", ["row-wise", "column-wise"], horizontal=True
                            )

                            axis = {"row-wise": 0, "column-wise": 1}[axis]
                            params = {
                                "fill_strategy": fill_strategy,
                                "axis": axis,
                                "cols_with_na": cols_with_na,
                            }
                        else:
                            params = {
                                "fill_strategy": fill_strategy,
                                "cols_with_na": cols_with_na,
                            }

                    with col3:
                        df = st.session_state[
                            "current_df"
                        ]  # Make sure you're using the same df
                        cols_with_na = df.columns[df.isnull().any()].tolist()

                        # Create a list of dicts for each column with missing values
                        data = []
                        for col in cols_with_na:
                            percent_missing = (
                                df[col].isnull().sum() / len(df[col])
                            ) * 100
                            status = "High" if percent_missing > 5.0 else "Low"
                            data.append(
                                {
                                    "Column": col,
                                    "Missing (%)": round(percent_missing, 2),
                                    "Severity": status,
                                }
                            )

                        # Create DataFrame
                        missing_df = pd.DataFrame(data)

                        def highlight_severity(val):
                            if val == "High":
                                return "color: rgba(240, 48, 48, 1); font-weight: bold;"
                            elif val == "Low":
                                return "color: rgba(45, 186, 89, 1); font-weight: bold;"
                            return ""

                        st.dataframe(
                            missing_df.style.format({"Missing (%)": "{:.2f}"}).applymap(
                                highlight_severity, subset=["Severity"]
                            )
                        )

                    if (
                        "High" in missing_df["Severity"].values
                        and fill_strategy == "Drop"
                    ):
                        st.info(
                            "NOTE: As a general guideline, if more than **5%** of the data is missing, "
                            "deleting (drop) rows with missing values may result in **considerable** information loss.",
                            icon=":material/info:",
                        )

                    if st.button("Apply", key="apply_missing_values"):
                        params = params
                        add_operation("missing_values", params)

                else:
                    st.info(
                        "There are no missing values in the DataFrame",
                        icon=":material/info:",
                    )

            with st.container(border=True):  # --- Remove Duplicates ---
                st.markdown("#### Remove Duplicate Rows")

                is_split = st.session_state.is_data_split

                if is_split:
                    st.warning(
                        "This operation can't be done after data split",
                        icon=":material/warning:",
                    )

                else:
                    if st.session_state["current_df"].duplicated(keep=False).sum() <= 0:
                        st.info(
                            "There are no duplicated rows in the DataFrame",
                            icon=":material/info:",
                        )
                    else:
                        col1, col2 = st.columns(2)

                        with col1:
                            display_duplicated = st.radio(
                                "Choose which duplicate rows to remove:",
                                ["All duplicates", "All but first occurrence"],
                                horizontal=True,
                            )
                        with col2:
                            if display_duplicated == "All duplicates":
                                duplicate_rows_sum = (
                                    st.session_state["current_df"]
                                    .duplicated(keep=False)
                                    .sum()
                                )
                                st.metric(
                                    label="Rows to remove:", value=duplicate_rows_sum
                                )
                                keep_first = False
                            elif display_duplicated == "All but first occurrence":
                                duplicate_rows_sum = (
                                    st.session_state["current_df"].duplicated().sum()
                                )
                                st.metric(
                                    label="Rows to remove:", value=duplicate_rows_sum
                                )
                                keep_first = True

                        with st.expander(
                            "Click here to display the duplicated rows..."
                        ):
                            if keep_first == False:
                                duplicate_rows = st.session_state["current_df"][
                                    st.session_state["current_df"].duplicated(
                                        keep=False
                                    )
                                ]
                                st.dataframe(duplicate_rows, use_container_width=True)
                            elif keep_first == True:
                                duplicate_rows = st.session_state["current_df"][
                                    st.session_state["current_df"].duplicated()
                                ]
                                st.dataframe(duplicate_rows, use_container_width=True)

                        if st.button("Apply", key="apply_remove_duplicates"):
                            params = {"keep_first": keep_first}
                            add_operation("remove_duplicates", params)

            with st.container(border=True):  # --- Datatype Convertion
                st.markdown("#### Datatype Convertion")

                if st.session_state["current_df"].isnull().sum().sum() <= 0:
                    available_columns = st.session_state["current_df"].columns.tolist()
                    data_types = ["int", "float", "str", "datetime", "category", "bool"]

                    col1, col2, col3, col4 = st.columns([3, 0.5, 1.5, 1.5])

                    with col1:
                        selected_column = st.selectbox(
                            "Select a column", available_columns
                        )

                    with col3:
                        selected_datatype = st.selectbox(
                            "Select a target data type", data_types
                        )

                    with col4:
                        current_dtype = st.session_state["current_df"][
                            selected_column
                        ].dtype
                        current_dtype = str(current_dtype)

                        st.metric(label="Current data type", value=current_dtype)

                    st.markdown("")
                    with st.expander("Display Data Type table..."):
                        dtypes_df = st.session_state["current_df"].dtypes
                        dtypes_df = dtypes_df.astype(str).reset_index()
                        dtypes_df.columns = ["Column", "Data Type"]

                        st.dataframe(dtypes_df, use_container_width=True)

                    if st.button("Apply", key="apply_dataype_convertion"):
                        params = {
                            "selected_column": selected_column,
                            "selected_datatype": selected_datatype,
                        }
                        add_operation("datatype_convertion", params)

                else:
                    st.warning(
                        "This operation is not possible if there are **missing values** in the DataFrame",
                        icon=":material/warning:",
                    )

            with st.container(border=True):  # --- Drop Columns ---
                st.markdown("#### Drop Columns")
                available_columns = st.session_state["current_df"].columns.tolist()
                selected_columns = st.multiselect(
                    "Select columns to drop", available_columns, key="drop_cols_select"
                )

                if st.button("Apply", key="apply_drop_columns"):
                    if selected_columns:
                        params = {"columns": selected_columns}
                        add_operation("drop_columns", params)
                    else:
                        st.warning(
                            "Please select columns to drop", icon=":material/warning:"
                        )

            with st.container(border=True):  # --- Remove Outliers ---
                st.markdown("#### Remove Outliers")
                if st.session_state.is_data_encoded == True:
                    st.warning(
                        "The DataFrame has been previously encoded. Undo the encoding to proceed.",
                        icon=":material/warning:",
                    )
                else:
                    num_cols = (
                        st.session_state["current_df"]
                        .select_dtypes(include=["int64", "float64"])
                        .columns.tolist()
                    )
                    if num_cols:
                        if st.session_state["current_df"].isnull().sum().sum() <= 0:
                            col1, col2, col3 = st.columns([2, 0.5, 4])

                            with col1:
                                numerical_cols = (
                                    st.session_state["current_df"]
                                    .select_dtypes(include=np.number)
                                    .columns.tolist()
                                )
                                selected_outlier_column = st.selectbox(
                                    "Select a numerical column",
                                    numerical_cols,
                                    key="outliers_select",
                                )

                            with col3:
                                outlier_removal_method = st.radio(
                                    "Select the outliers removal method",
                                    ["IQR", "Percentage", "Isolation Forest"],
                                    horizontal=True,
                                )

                            if outlier_removal_method == "IQR":
                                params = {
                                    "selected_outlier_column": selected_outlier_column,
                                    "outlier_removal_method": outlier_removal_method,
                                }

                            elif outlier_removal_method == "Isolation Forest":
                                params = {
                                    "selected_outlier_column": selected_outlier_column,
                                    "outlier_removal_method": outlier_removal_method,
                                }

                            elif outlier_removal_method == "Percentage":
                                top_percentage = st.slider(
                                    "Percentage to remove from the :red[**top**]",
                                    0.0,
                                    0.5,
                                    0.01,
                                    0.01,
                                )
                                bottom_percentage = st.slider(
                                    "Percentage to remove from the :red[**bottom**]",
                                    0.0,
                                    0.5,
                                    0.01,
                                    0.01,
                                )

                                params = {
                                    "selected_outlier_column": selected_outlier_column,
                                    "top_percentage": top_percentage,
                                    "bottom_percentage": bottom_percentage,
                                    "outlier_removal_method": outlier_removal_method,
                                }

                            if st.button("Apply", key="apply_remove_outliers"):
                                add_operation("remove_outliers", params)

                            if st.session_state.is_data_split == False:
                                with st.expander(
                                    "Click to display distribution histogram"
                                ):
                                    outliers_preview = px.histogram(
                                        st.session_state["current_df"],
                                        x=st.session_state["current_df"][
                                            selected_outlier_column
                                        ],
                                        color_discrete_sequence=["#f03a11"],
                                        title=f"Distribution of {selected_outlier_column}",
                                    )
                                    st.plotly_chart(
                                        outliers_preview, key="outliers_histogram"
                                    )

                            if (
                                st.session_state.is_data_split
                                and st.session_state["validation_df"] is None
                            ):
                                with st.expander(
                                    "Click to display distribution histograms"
                                ):

                                    # Prepare combined DataFrame for Train + Test
                                    df_train = st.session_state["current_df"].copy()
                                    df_test = st.session_state["test_df"].copy()

                                    df_train["Set"] = "Train"
                                    df_test["Set"] = "Test"
                                    df_combined = pd.concat([df_train, df_test])

                                    normalize_hist = st.toggle(
                                        "Normalize histogram",
                                        value=False,
                                        key="norm_train_test",
                                    )
                                    hist_norm = (
                                        "probability density"
                                        if normalize_hist
                                        else None
                                    )

                                    combined_hist = px.histogram(
                                        df_combined,
                                        x=selected_outlier_column,
                                        color="Set",
                                        color_discrete_map={
                                            "Train": "#1f77b4",
                                            "Test": "#f03a11",
                                        },
                                        barmode="overlay",  # or "group" for side-by-side
                                        histnorm=hist_norm,
                                        title=f"Train vs Test Distribution ({selected_outlier_column})",
                                    )
                                    st.plotly_chart(
                                        combined_hist,
                                        key="train_test_combined_histogram_1",
                                    )

                            if (
                                st.session_state.is_data_split
                                and st.session_state["validation_df"] is not None
                            ):
                                with st.expander(
                                    "Click to display distribution histograms"
                                ):

                                    # Create copies and tag each with its set label
                                    df_train = st.session_state["current_df"].copy()
                                    df_val = st.session_state["validation_df"].copy()
                                    df_test = st.session_state["test_df"].copy()

                                    df_train["Set"] = "Train"
                                    df_val["Set"] = "Validation"
                                    df_test["Set"] = "Test"

                                    # Concatenate all into one DataFrame
                                    df_combined = pd.concat([df_train, df_val, df_test])

                                    normalize_hist = st.toggle(
                                        "Normalize histogram",
                                        value=False,
                                        key="norm_train_val_test",
                                    )
                                    hist_norm = (
                                        "probability density"
                                        if normalize_hist
                                        else None
                                    )
                                    combined_hist = px.histogram(
                                        df_combined,
                                        x=selected_outlier_column,
                                        color="Set",
                                        color_discrete_map={
                                            "Train": "#1f77b4",
                                            "Validation": "#f03a11",
                                            "Test": "#2ca02c",
                                        },
                                        barmode="overlay",  # Or use "group" for side-by-side comparison
                                        # barmode="group",
                                        histnorm=hist_norm,
                                        title=f"Distribution of {selected_outlier_column} (Train, Validation, Test)",
                                    )
                                    st.plotly_chart(
                                        combined_hist,
                                        key="all_sets_combined_histogram_norm",
                                    )

                            else:
                                pass
                        else:
                            st.warning(
                                "This operation is not possible if there are **missing values** in the DataFrame",
                                icon=":material/warning:",
                            )
                    else:
                        st.warning(
                            "No numeric columns to remove outliers from",
                            icon=":material/warning:",
                        )

        with tab3:  # --- Exploratory Data Analysis ---
            st.subheader("📊 Exploratory Data Analysis")
            with st.container(border=True):  # --- Statistics Summary ---
                st.markdown("#### Statistics Summary")
                summary_display = st.radio(
                    "Select the way you would like to display the DataFrame statistics",
                    ["Single Column", "Full Table"],
                )

                if summary_display == "Single Column":
                    # st.markdown("#### Column Statistics")
                    df = st.session_state["current_df"]
                    data_description = st.session_state["current_df"].describe(
                        include="all"
                    )
                    selected_column = st.selectbox(
                        "Select a column to see the whole description",
                        data_description.columns,
                    )
                    st.markdown(f"Data from :red[**{selected_column}**] column:")
                    st.dataframe(
                        data_description[[selected_column]].T, use_container_width=True
                    )

                elif summary_display == "Full Table":
                    # st.markdown("#### Statistics table")
                    df = st.session_state["current_df"]
                    # st.caption("Here are the `count` and `unique` rows from the dataframe's description table.")
                    data_description = st.session_state["current_df"].describe(
                        include="all"
                    )
                    # st.dataframe(data_description.iloc[:2,:], use_container_width=True)

                    st.dataframe(data_description, use_container_width=True)
                    st.caption(
                        """
                                This data is useful to notice any missing values or to define if a column with 
                                categorical values is suitable for the creation of dummy variables or not.
                                """
                    )

            with st.container(border=True):  # --- Display Visualization ---
                st.markdown("#### Data Visualization")
                num_cols = (
                    st.session_state["current_df"]
                    .select_dtypes(include=np.number)
                    .columns.tolist()
                )
                cat_cols = (
                    st.session_state["current_df"]
                    .select_dtypes(include=["object"])
                    .columns.tolist()
                )

                # All cols
                all_cols = st.session_state["current_df"].columns.tolist()

                plot_type = st.selectbox(
                    "Select the type of graph you would like to display",
                    (
                        "Histogram",
                        "KDE",
                        "Box plot",
                        "Violin plot",
                        "Scatter plot",
                        "Heatmap",
                        "Line plot",
                    ),
                )

                match plot_type:
                    case "Histogram":
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            num_or_cat = st.radio(
                                "Feature type",
                                ["Numerical", "Categorical"],
                                horizontal=True,
                            )
                        with col2:
                            if num_or_cat == "Numerical":
                                hist_col = st.selectbox(
                                    "Select a numerical column", num_cols
                                )

                            else:
                                hist_col = st.selectbox(
                                    "Select a categorical column", cat_cols
                                )

                        with col3:
                            if st.session_state.is_data_split == False:
                                selected_color = st.selectbox(
                                    "Choose a color", ("Cyan", "Violet", "Blue", "Pink")
                                )

                                match selected_color:
                                    case "Cyan":
                                        selected_color = "#63F5EF"
                                    case "Violet":
                                        selected_color = "#A56CC1"
                                    case "Blue":
                                        selected_color = "#0073fa"
                                    case "Pink":
                                        selected_color = "#F66095"

                        if st.session_state.is_data_split == False:
                            if num_or_cat == "Numerical" and num_cols:
                                fig_cat_hist = px.histogram(
                                    st.session_state["current_df"],
                                    x=st.session_state["current_df"][hist_col],
                                    category_orders=dict(x=[hist_col]),
                                    color_discrete_sequence=[selected_color],
                                    opacity=0.9,
                                )
                                st.plotly_chart(fig_cat_hist, key="fig_cat_hist_num")

                            elif num_or_cat == "Numerical" and not num_cols:
                                st.info(
                                    "The DataFrame does not contain any numerical columns.",
                                    icon=":material/info:",
                                )

                            elif num_or_cat == "Categorical" and cat_cols:
                                fig_cat_hist = px.histogram(
                                    st.session_state["current_df"],
                                    x=st.session_state["current_df"][hist_col],
                                    category_orders=dict(x=[hist_col]),
                                    color_discrete_sequence=[selected_color],
                                    opacity=0.9,
                                )
                                st.plotly_chart(fig_cat_hist, key="fig_cat_hist_cat")
                            elif num_or_cat == "Categorical" and not cat_cols:
                                st.info(
                                    "The DataFrame does not contain any categorical columns.",
                                    icon=":material/info:",
                                )

                        if (
                            st.session_state.is_data_split
                            and st.session_state["validation_df"] is None
                        ):
                            df_train = st.session_state["current_df"].copy()
                            df_test = st.session_state["test_df"].copy()

                            df_train["Set"] = "Train"
                            df_test["Set"] = "Test"
                            df_combined = pd.concat([df_train, df_test])

                            normalize_hist = st.toggle(
                                "Normalize histogram",
                                value=False,
                                key="norm_train_test_split",
                            )
                            hist_norm = (
                                "probability density" if normalize_hist else None
                            )

                            if num_or_cat == "Numerical" and num_cols:
                                combined_hist = px.histogram(
                                    df_combined,
                                    x=df_combined[hist_col],
                                    color="Set",
                                    color_discrete_map={
                                        "Train": "#1f77b4",
                                        "Test": "#f03a11",
                                    },
                                    barmode="overlay",  # or "group" for side-by-side
                                    histnorm=hist_norm,
                                    title=f"Train vs Test Distribution ({hist_col})",
                                )
                                st.plotly_chart(
                                    combined_hist,
                                    key="train_test_combined_histogram_num",
                                )

                            elif num_or_cat == "Numerical" and not num_cols:
                                st.info(
                                    "The DataFrame does not contain any numerical columns.",
                                    icon=":material/info:",
                                )

                            elif num_or_cat == "Categorical" and cat_cols:
                                combined_hist = px.histogram(
                                    df_combined,
                                    x=df_combined[hist_col],
                                    color="Set",
                                    color_discrete_map={
                                        "Train": "#1f77b4",
                                        "Test": "#f03a11",
                                    },
                                    barmode="overlay",  # or "group" for side-by-side
                                    histnorm=hist_norm,
                                    title=f"Train vs Test Distribution ({hist_col})",
                                )
                                st.plotly_chart(
                                    combined_hist,
                                    key="train_test_combined_histogram_cat",
                                )

                            elif num_or_cat == "Categorical" and not cat_cols:
                                st.info(
                                    "The DataFrame does not contain any categorical columns.",
                                    icon=":material/info:",
                                )

                        if (
                            st.session_state.is_data_split
                            and st.session_state["validation_df"] is not None
                        ):
                            df_train = st.session_state["current_df"].copy()
                            df_val = st.session_state["validation_df"].copy()
                            df_test = st.session_state["test_df"].copy()

                            df_train["Set"] = "Train"
                            df_val["Set"] = "Validation"
                            df_test["Set"] = "Test"
                            df_combined = pd.concat([df_train, df_val, df_test])

                            normalize_hist = st.toggle(
                                "Normalize histogram",
                                value=False,
                                key="norm_train_test_val_split",
                            )
                            hist_norm = (
                                "probability density" if normalize_hist else None
                            )

                            if num_or_cat == "Numerical" and num_cols:
                                combined_hist = px.histogram(
                                    df_combined,
                                    x=df_combined[hist_col],
                                    color="Set",
                                    color_discrete_map={
                                        "Train": "#1f77b4",
                                        "Validation": "#f03a11",
                                        "Test": "#2ca02c",
                                    },
                                    barmode="overlay",  # or "group" for side-by-side
                                    histnorm=hist_norm,
                                    title=f"Train vs Test Distribution ({hist_col})",
                                )
                                st.plotly_chart(
                                    combined_hist, key="train_val_test_combined_hist"
                                )

                            elif num_or_cat == "Numerical" and not num_cols:
                                st.info(
                                    "The DataFrame does not contain any numerical columns.",
                                    icon=":material/info:",
                                )

                            elif num_or_cat == "Categorical" and cat_cols:
                                combined_hist = px.histogram(
                                    df_combined,
                                    x=df_combined[hist_col],
                                    color="Set",
                                    color_discrete_map={
                                        "Train": "#1f77b4",
                                        "Validation": "#f03a11",
                                        "Test": "#2ca02c",
                                    },
                                    barmode="overlay",  # or "group" for side-by-side
                                    histnorm=hist_norm,
                                    title=f"Train vs Test Distribution ({hist_col})",
                                )
                                st.plotly_chart(
                                    combined_hist, key="train_val_test_combined_hist"
                                )

                            elif num_or_cat == "Categorical" and not cat_cols:
                                st.info(
                                    "The DataFrame does not contain any categorical columns.",
                                    icon=":material/info:",
                                )

                    case "KDE":
                        if st.session_state["current_df"].isnull().sum().sum() <= 0:
                            col1, col2 = st.columns(2)

                            with col1:
                                kde_col = st.selectbox(
                                    "Select a numerical column", num_cols
                                )
                            with col2:
                                if st.session_state.is_data_split == False:
                                    selected_color = st.selectbox(
                                        "Choose a color",
                                        ("Cyan", "Violet", "Blue", "Pink"),
                                    )

                                    match selected_color:
                                        case "Cyan":
                                            selected_color = "#63F5EF"
                                        case "Violet":
                                            selected_color = "#A56CC1"
                                        case "Blue":
                                            selected_color = "#0073fa"
                                        case "Pink":
                                            selected_color = "#F66095"

                            if st.session_state.is_data_split == False:
                                hist_data = [df[kde_col]]
                                group_labels = [kde_col]
                                colors = [selected_color]
                                fig_kde = ff.create_distplot(
                                    hist_data,
                                    group_labels,
                                    colors=colors,
                                    show_hist=False,
                                    show_rug=True,
                                )
                                st.plotly_chart(fig_kde, key="fig_kde")

                            if (
                                st.session_state.is_data_split
                                and st.session_state["validation_df"] is None
                            ):

                                df_train = st.session_state["current_df"].copy()
                                df_test = st.session_state["test_df"].copy()

                                hist_data = [df_train[kde_col], df_test[kde_col]]
                                group_labels = ["Train", "Test"]
                                fig_kde_split = ff.create_distplot(
                                    hist_data,
                                    group_labels,
                                    show_hist=False,
                                    show_rug=True,
                                )
                                st.plotly_chart(fig_kde_split, key="fig_kde_split")

                            if (
                                st.session_state.is_data_split
                                and st.session_state["validation_df"] is not None
                            ):

                                df_train = st.session_state["current_df"].copy()
                                df_val = st.session_state["validation_df"].copy()
                                df_test = st.session_state["test_df"].copy()

                                hist_data = [
                                    df_train[kde_col],
                                    df_val[kde_col],
                                    df_test[kde_col],
                                ]
                                group_labels = ["Train", "Validation", "Test"]
                                fig_kde_split = ff.create_distplot(
                                    hist_data,
                                    group_labels,
                                    show_hist=False,
                                    show_rug=True,
                                )
                                st.plotly_chart(fig_kde_split, key="fig_kde_split_val")

                        else:
                            st.warning(
                                "This operation is not possible if there are **missing values** in the DataFrame",
                                icon=":material/warning:",
                            )

                    case "Box plot":
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            box_col = st.selectbox("Select a column", num_cols)

                        with col2:
                            box_points = st.selectbox(
                                "Boxpoints",
                                ("all", False, "suspectedoutliers", "outliers"),
                            )

                        with col3:
                            quart_method = st.selectbox(
                                "Quartile", ("linear", "inclusive", "exclusive")
                            )

                        with col4:
                            color_list = (
                                "lightseagreen",
                                "aliceblue",
                                "fuchsia",
                                "forestgreen",
                                "plum",
                                "orangered",
                                "yellowgreen",
                            )
                            color = st.selectbox("Choose a color", color_list)

                        if st.session_state.is_data_split == False:
                            fig_box_plot = go.Figure()
                            fig_box_plot.add_trace(
                                go.Box(
                                    x=df[box_col],
                                    jitter=0.3,
                                    quartilemethod=quart_method,
                                    boxpoints=box_points,
                                    marker_color=color,
                                )
                            )

                            fig_box_plot.update_layout(
                                height=400,  # Adjust as needed
                                # width=800    # Adjust as needed
                            )
                            st.plotly_chart(fig_box_plot, key="fig_box_plot")

                        if (
                            st.session_state.is_data_split
                            and st.session_state["validation_df"] is None
                        ):

                            df_train = st.session_state["current_df"].copy()
                            df_test = st.session_state["test_df"].copy()

                            # "Train": "#1f77b4",
                            # "Validation": "#f03a11",
                            # "Test": "#2ca02c"

                            fig_box_plot = go.Figure()
                            fig_box_plot.add_trace(
                                go.Box(
                                    x=df_train[box_col],
                                    jitter=0.3,
                                    quartilemethod=quart_method,
                                    boxpoints=box_points,
                                    marker_color="#1f77b4",
                                )
                            )
                            fig_box_plot.add_trace(
                                go.Box(
                                    x=df_test[box_col],
                                    jitter=0.3,
                                    quartilemethod=quart_method,
                                    boxpoints=box_points,
                                    marker_color="#2ca02c",
                                )
                            )

                            fig_box_plot.update_layout(
                                height=400,  # Adjust as needed
                                # width=800    # Adjust as needed
                            )
                            st.plotly_chart(fig_box_plot, key="fig_box_plot_split")

                        if (
                            st.session_state.is_data_split
                            and st.session_state["validation_df"] is not None
                        ):

                            df_train = st.session_state["current_df"].copy()
                            df_val = st.session_state["validation_df"].copy()
                            df_test = st.session_state["test_df"].copy()

                            # "Train": "#1f77b4",
                            # "Validation": "#f03a11",
                            # "Test": "#2ca02c"

                            fig_box_plot = go.Figure()
                            fig_box_plot.add_trace(
                                go.Box(
                                    x=df_train[box_col],
                                    jitter=0.3,
                                    quartilemethod=quart_method,
                                    boxpoints=box_points,
                                    marker_color="#1f77b4",
                                )
                            )
                            fig_box_plot.add_trace(
                                go.Box(
                                    x=df_val[box_col],
                                    jitter=0.3,
                                    quartilemethod=quart_method,
                                    boxpoints=box_points,
                                    marker_color="#f03a11",
                                )
                            )
                            fig_box_plot.add_trace(
                                go.Box(
                                    x=df_test[box_col],
                                    jitter=0.3,
                                    quartilemethod=quart_method,
                                    boxpoints=box_points,
                                    marker_color="#2ca02c",
                                )
                            )

                            fig_box_plot.update_layout(
                                height=400,  # Adjust as needed
                                # width=800    # Adjust as needed
                            )
                            st.plotly_chart(fig_box_plot, key="fig_box_plot_split_val")

                    case "Violin plot":
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            violin_col = st.selectbox(
                                "Select a column", cat_cols, key="violin_col_1"
                            )

                        with col2:
                            y_col = st.selectbox(
                                "Select a column", num_cols, key="violin_col_2"
                            )

                            # box_points = st.selectbox("Boxpoints", ("all", False, "suspectedoutliers", "outliers"))

                        # with col3:
                        # quart_method = st.selectbox("Quartile",("linear", "inclusive", "exclusive"))

                        with col4:
                            color_list = (
                                "lightseagreen",
                                "aliceblue",
                                "fuchsia",
                                "forestgreen",
                                "plum",
                                "orangered",
                                "yellowgreen",
                            )
                            color = st.selectbox("Choose a color", color_list)

                        if st.session_state.is_data_split == False:
                            fig_violin = go.Figure()

                            fig_violin.add_trace(
                                go.Violin(
                                    x=df[violin_col],
                                    y=df[y_col],
                                    # name=day,
                                    box_visible=True,
                                    meanline_visible=True,
                                )
                            )

                            st.plotly_chart(fig_violin, key="fig_violin_plot")

                        #     fig_violin = px.violin(df,

                        #                     y=y_col,
                        #                     violinmode='overlay', # draw violins on top of each other
                        #                     color=color,
                        #                     # default violinmode is 'group' as in example above
                        #                     hover_data=df.columns)
                        #     st.plotly_chart(fig_violin, key='fig_violin_plot')

                        # #********************************
                        #     fig_box_plot = go.Figure()
                        #     fig_box_plot.add_trace(go.Box(
                        #         x=df[box_col],
                        #         jitter=0.3,
                        #         quartilemethod=quart_method,
                        #         boxpoints= box_points,
                        #         marker_color = color
                        #     ))

                        #     fig_box_plot.update_layout(
                        #         height=400,  # Adjust as needed
                        #         # width=800    # Adjust as needed
                        #     )
                        #     st.plotly_chart(fig_box_plot, key='fig_box_plot')

                    case "Scatter plot":
                        if st.session_state.is_data_split == False:
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                scatter_col_1 = st.selectbox(
                                    "Select X column", all_cols
                                )
                                # scatter_col_1 = st.selectbox("Select X column", num_cols)

                                trend_line = st.toggle("Regression Line")
                                ols_line = "ols" if trend_line else None

                            with col2:
                                available_y_cols = [
                                    col for col in all_cols if col != scatter_col_1
                                ]
                                scatter_col_2 = st.selectbox(
                                    "Select Y column", available_y_cols
                                )

                                # available_y_cols = [col for col in num_cols if col != scatter_col_1]
                                # scatter_col_2 = st.selectbox("Select Y column", available_y_cols)

                            with col3:
                                color_style = st.radio(
                                    "Color style",
                                    ["Solid", "Gradient"],
                                    horizontal=True,
                                )

                            with col4:
                                if color_style == "Gradient":
                                    gradient_scales = (
                                        "electric",
                                        "jet",
                                        "plasma",
                                        "inferno",
                                        "hot",
                                        "viridis",
                                    )
                                    gradient = st.selectbox(
                                        "Select gradient", gradient_scales
                                    )

                                    fig_scatter = px.scatter(
                                        df,
                                        x=scatter_col_1,
                                        y=scatter_col_2,
                                        color=df[scatter_col_2],
                                        color_continuous_scale=gradient,
                                        trendline=ols_line,
                                    )
                                    fig_scatter.update_layout(height=600, width=600)

                                elif color_style == "Solid":
                                    color_list = (
                                        "lightseagreen",
                                        "aliceblue",
                                        "fuchsia",
                                        "forestgreen",
                                        "plum",
                                        "orangered",
                                        "yellowgreen",
                                    )
                                    selected_color = st.selectbox(
                                        "Choose a color", color_list
                                    )

                                    fig_scatter = px.scatter(
                                        df,
                                        x=scatter_col_1,
                                        y=scatter_col_2,
                                        color=df[scatter_col_2],
                                        trendline=ols_line,
                                    )
                                    fig_scatter.update_traces(
                                        marker=dict(color=selected_color)
                                    )
                                    fig_scatter.update_layout(height=600, width=600)

                            st.plotly_chart(fig_scatter, key="fig_scatter")

                        elif (
                            st.session_state.is_data_split
                            and st.session_state["validation_df"] is None
                        ):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                scatter_col_1 = st.selectbox(
                                    "Select X column", num_cols
                                )

                            with col2:
                                available_y_cols = [
                                    col for col in num_cols if col != scatter_col_1
                                ]
                                scatter_col_2 = st.selectbox(
                                    "Select Y column", available_y_cols
                                )

                            with col3:
                                trend_line = st.toggle("Regression Line")
                                ols_line = "ols" if trend_line else None

                            df_train = st.session_state["current_df"].copy()
                            df_test = st.session_state["test_df"].copy()

                            df_train["Set"] = "Train"
                            df_test["Set"] = "Test"

                            df_combined = pd.concat(
                                [df_train, df_test], ignore_index=True
                            )

                            fig_scatter = px.scatter(
                                df_combined,
                                x=scatter_col_1,
                                y=scatter_col_2,
                                color="Set",
                                title="Scatter Plot: Train & Test Sets",
                                color_discrete_map={
                                    "Train": "#1f77b4",
                                    "Test": "#2ca02c",
                                },
                                trendline=ols_line,
                            )

                            fig_scatter.update_layout(
                                height=600,
                                width=800,
                                xaxis_title=scatter_col_1,
                                yaxis_title=scatter_col_2,
                                legend_title_text="Dataset Type",
                            )

                            st.plotly_chart(
                                fig_scatter, key="fig_scatter_combined_sets"
                            )

                        elif (
                            st.session_state.is_data_split
                            and st.session_state["validation_df"] is not None
                        ):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                scatter_col_1 = st.selectbox(
                                    "Select X column", num_cols
                                )

                            with col2:
                                available_y_cols = [
                                    col for col in num_cols if col != scatter_col_1
                                ]
                                scatter_col_2 = st.selectbox(
                                    "Select Y column", available_y_cols
                                )

                            with col3:
                                trend_line = st.toggle("Regression Line")
                                ols_line = "ols" if trend_line else None

                            # 1. Add an identifier column to each dataframe
                            df_train = st.session_state["current_df"].copy()
                            df_val = st.session_state["validation_df"].copy()
                            df_test = st.session_state["test_df"].copy()

                            df_train["Set"] = "Train"
                            df_val["Set"] = "Validation"
                            df_test["Set"] = "Test"

                            df_combined = pd.concat(
                                [df_train, df_val, df_test], ignore_index=True
                            )

                            fig_scatter = px.scatter(
                                df_combined,
                                x=scatter_col_1,
                                y=scatter_col_2,
                                color="Set",  # <-- This is the key line to differentiate colors
                                title="Scatter Plot: Train, Test, & Validation Sets",
                                color_discrete_map={
                                    "Train": "#1f77b4",
                                    "Validation": "#f03a11",
                                    "Test": "#2ca02c",
                                },
                                trendline=ols_line,
                            )

                            fig_scatter.update_layout(
                                height=600,
                                width=800,
                                xaxis_title=scatter_col_1,
                                yaxis_title=scatter_col_2,
                                legend_title_text="Dataset Type",
                            )

                            st.plotly_chart(
                                fig_scatter, key="fig_scatter_combined_sets_val"
                            )

                    case "Heatmap":
                        col1, col2, col3, col4 = st.columns([3, 3, 2, 2])

                        with col1:
                            heat_col_1 = st.selectbox("Select X column", num_cols)

                            # text_on_off = st.toggle("Text on")
                            # text_on = True if text_on_off else False

                        with col2:
                            heat_col_2 = st.selectbox("Select Y column", num_cols)

                        with col3:
                            color_list = (
                                "electric",
                                "jet",
                                "plasma",
                                "blackbody",
                                "inferno",
                                "hot",
                            )
                            color = st.selectbox("Choose a color", color_list)

                        with col4:
                            # text_on = st.selectbox("Text",(False, True))
                            text_on_off = st.toggle("Text on")
                            text_on = True if text_on_off else False

                        if st.session_state.is_data_split == False:
                            fig_heatmap = px.density_heatmap(
                                df,
                                x=heat_col_1,
                                y=heat_col_2,
                                text_auto=text_on,
                                color_continuous_scale=color,
                            )
                            fig_heatmap.update_layout(height=600, width=600)
                            st.plotly_chart(fig_heatmap, key="fig_heatmap")

                        elif (
                            st.session_state.is_data_split
                            and st.session_state["validation_df"] is None
                        ):
                            df_train = st.session_state["current_df"].copy()
                            df_test = st.session_state["test_df"].copy()

                            df_combined = pd.concat(
                                [df_train, df_test], ignore_index=True
                            )

                            fig_heatmap = px.density_heatmap(
                                df_combined,
                                x=heat_col_1,
                                y=heat_col_2,
                                text_auto=text_on,
                                color_continuous_scale=color,
                            )
                            fig_heatmap.update_layout(height=600, width=600)
                            st.plotly_chart(fig_heatmap, key="fig_heatmap_split")

                        elif (
                            st.session_state.is_data_split
                            and st.session_state["validation_df"] is not None
                        ):
                            df_train = st.session_state["current_df"].copy()
                            df_val = st.session_state["validation_df"].copy()
                            df_test = st.session_state["test_df"].copy()

                            df_combined = pd.concat(
                                [df_train, df_val, df_test], ignore_index=True
                            )

                            fig_heatmap = px.density_heatmap(
                                df_combined,
                                x=heat_col_1,
                                y=heat_col_2,
                                text_auto=text_on,
                                color_continuous_scale=color,
                            )
                            fig_heatmap.update_layout(height=600, width=600)
                            st.plotly_chart(fig_heatmap, key="fig_heatmap_split_val")

                    case "Line plot":  # To be developed whith time series data
                        time_col = (
                            st.session_state["current_df"]
                            .select_dtypes(include=["datetime"])
                            .columns.tolist()
                        )
                        if time_col:
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                line_col_1 = st.selectbox("Select X column", time_col)

                            with col2:
                                available_y_cols = [
                                    col for col in num_cols if col != line_col_1
                                ]
                                line_col_2 = st.selectbox(
                                    "Select Y column", available_y_cols
                                )

                            with col3:
                                color_list = (
                                    "aliceblue",
                                    "lightseagreen",
                                    "fuchsia",
                                    "forestgreen",
                                    "plum",
                                    "orangered",
                                    "yellowgreen",
                                )
                                selected_color = st.selectbox(
                                    "Choose a color", color_list
                                )

                            fig_line = px.line(
                                df, x=line_col_1, y=line_col_2, markers=True
                            )
                            # fig_line = px.line(df, x=line_col_1, y=line_col_2, color=selected_color, markers=True)
                            fig_line.update_traces(marker=dict(color=selected_color))
                            st.plotly_chart(fig_line, key="fig_line")
                        else:
                            st.info(
                                "The DataFrame does not contain any 'datatime' column to create a line plot.",
                                icon=":material/info:",
                            )

        with tab4:  # --- Data Splitting ---
            st.subheader("✂️ Train-Test Split")
            with st.container(border=True):  # --- Train -Test Splitting ---
                st.markdown("#### Train-Test Split")
                is_split = st.session_state.is_data_split

                if is_split:
                    st.info("The data has already been split", icon=":material/info:")
                else:
                    if st.session_state["current_df"].isnull().sum().sum() <= 0:
                        with st.container():
                            col1, col2, col3 = st.columns([6, 0.5, 3])
                            with col1:
                                validation_split = st.toggle(
                                    "Add a validation set",
                                    value=True,
                                    key="add_validation",
                                )

                                # Initial values
                                default_train = 80
                                default_val = 10
                                default_test = 20

                                if validation_split:
                                    default_train = 70
                                    train_size = st.slider(
                                        "Train Set (%)", 0, 100, default_train
                                    )

                                    remaining_after_train = 100 - train_size

                                    # val_size = st.slider("Validation Set (%)", 0, remaining_after_train, min(default_val, remaining_after_train))
                                    validation_size = st.slider(
                                        "Validation Set (%)", 0, 100, default_val
                                    )

                                    value_test_size = 100 - train_size - validation_size
                                    test_size = st.slider(
                                        "Test Set (%)",
                                        min_value=0,
                                        max_value=100,
                                        value=value_test_size,
                                        disabled=True,
                                    )

                                    test_size = test_size / 100
                                    validation_size = validation_size / 100

                                else:
                                    train_size = st.slider(
                                        "Train Set (%)", 0, 100, default_train
                                    )
                                    remaining_after_train = 100 - train_size
                                    test_size = st.slider(
                                        "Test Set (%)",
                                        min_value=0,
                                        max_value=100,
                                        value=remaining_after_train,
                                        disabled=True,
                                    )
                                    test_size = test_size / 100

                                shuffle_split = st.toggle(
                                    "Shuffle data", value=True, key="shuffle"
                                )

                            with col3:
                                random_state = st.number_input(
                                    "Random State", min_value=0, step=1, value=42
                                )

                                stratify_col = st.selectbox(
                                    "Stratify by (optional)",
                                    [None]
                                    + st.session_state["current_df"].columns.tolist(),
                                )

                                num_bins_for_stratify = st.number_input(
                                    "Bins to stratify", min_value=0, step=1, value=10
                                )

                        if validation_split:
                            if st.button("Split Data"):
                                params = {
                                    "test_size": test_size,
                                    "validation_size": validation_size,
                                    "random_state": random_state,
                                    "stratify_col": (
                                        None if stratify_col == "None" else stratify_col
                                    ),
                                    "shuffle": shuffle_split,
                                    "stratify_num_bins": num_bins_for_stratify,
                                }
                                add_operation("train_test_val_split", params)
                        else:
                            if st.button("Split Data"):
                                params = {
                                    "test_size": test_size,
                                    "random_state": random_state,
                                    "stratify_col": (
                                        None if stratify_col == "None" else stratify_col
                                    ),
                                    "shuffle": shuffle_split,
                                    "stratify_num_bins": num_bins_for_stratify,
                                }
                                add_operation("train_test_split", params)

                    else:
                        st.warning(
                            "This operation is not possible if there are **missing values** in the DataFrame",
                            icon=":material/warning:",
                        )

        with tab5:  # --- Data Transformation ---
            st.subheader("🔄 Data Transformation")
            with st.container(border=True):  # --- Power Transformation ---
                st.markdown("#### Power Transformation")

                if st.session_state.is_data_encoded == False:
                    if st.session_state["current_df"].isnull().sum().sum() <= 0:
                        num_cols = (
                            st.session_state["current_df"]
                            .select_dtypes(include=["int64", "float64"])
                            .columns.tolist()
                        )
                        if num_cols:
                            col1, col2, col3 = st.columns([2, 0.5, 4])

                            with col1:
                                col_to_transform = st.selectbox(
                                    "Select a numerical column to transform",
                                    num_cols,
                                    key="transform_select",
                                )

                            with col3:
                                transform_method = st.radio(
                                    "Select the Transform Method",
                                    [
                                        "Log Transform",
                                        "Box-Cox Transform",
                                        "Yeo-Johnson Transform",
                                    ],
                                    horizontal=True,
                                )

                            if st.button("Apply", key="apply_transform"):
                                params = {
                                    "transform_method": transform_method,
                                    "num_cols": num_cols,
                                    "col_to_transform": col_to_transform,
                                }
                                add_operation("transform_columns", params)

                            if st.session_state.is_data_split == False:
                                with st.expander(
                                    "Click to display distribution histogram"
                                ):
                                    transform_preview = px.histogram(
                                        st.session_state["current_df"],
                                        x=st.session_state["current_df"][
                                            col_to_transform
                                        ],
                                        color_discrete_sequence=["#f03a11"],
                                        title=f"Distribution of {col_to_transform}",
                                    )
                                    st.plotly_chart(
                                        transform_preview,
                                        key="transformation_histogram",
                                    )

                            if (
                                st.session_state.is_data_split
                                and st.session_state["validation_df"] is None
                            ):
                                with st.expander(
                                    "Click to display distribution histograms"
                                ):

                                    # Prepare combined DataFrame for Train + Test
                                    df_train = st.session_state["current_df"].copy()
                                    df_test = st.session_state["test_df"].copy()

                                    df_train["Set"] = "Train"
                                    df_test["Set"] = "Test"
                                    df_combined = pd.concat([df_train, df_test])

                                    normalize_hist = st.toggle(
                                        "Normalize histogram",
                                        value=False,
                                        key="norm_train_test_hist",
                                    )
                                    hist_norm = (
                                        "probability density"
                                        if normalize_hist
                                        else None
                                    )

                                    transform_preview = px.histogram(
                                        df_combined,
                                        x=col_to_transform,
                                        color="Set",
                                        color_discrete_map={
                                            "Train": "#1f77b4",
                                            "Test": "#f03a11",
                                        },
                                        barmode="overlay",  # or "group" for side-by-side
                                        histnorm=hist_norm,
                                        title=f"Distribution of {col_to_transform}",
                                    )
                                    st.plotly_chart(
                                        transform_preview,
                                        key="transformation_train_test_histogram",
                                    )

                            if (
                                st.session_state.is_data_split
                                and st.session_state["validation_df"] is not None
                            ):
                                with st.expander(
                                    "Click to display distribution histograms"
                                ):

                                    # Prepare combined DataFrame for Train + Test
                                    df_train = st.session_state["current_df"].copy()
                                    df_val = st.session_state["validation_df"].copy()
                                    df_test = st.session_state["test_df"].copy()

                                    df_train["Set"] = "Train"
                                    df_val["Set"] = "Validation"
                                    df_test["Set"] = "Test"
                                    df_combined = pd.concat([df_train, df_val, df_test])

                                    normalize_hist = st.toggle(
                                        "Normalize histogram",
                                        value=False,
                                        key="norm_train_test_val_hist",
                                    )
                                    hist_norm = (
                                        "probability density"
                                        if normalize_hist
                                        else None
                                    )

                                    transform_preview = px.histogram(
                                        df_combined,
                                        x=col_to_transform,
                                        color="Set",
                                        color_discrete_map={
                                            "Train": "#1f77b4",
                                            "Validation": "#f03a11",
                                            "Test": "#2ca02c",
                                        },
                                        barmode="overlay",  # or "group" for side-by-side
                                        histnorm=hist_norm,
                                        title=f"Distribution of {col_to_transform}",
                                    )
                                    st.plotly_chart(
                                        transform_preview,
                                        key="transformation_train_test_val_histogram",
                                    )

                            else:
                                pass
                        else:
                            st.warning(
                                "No numeric columns to transform",
                                icon=":material/warning:",
                            )
                    else:
                        st.warning(
                            "This operation is not possible if there are **missing values** in the DataFrame",
                            icon=":material/warning:",
                        )
                else:
                    st.warning(
                        "This operation is not possible if the DataFrame was previously encoded",
                        icon=":material/warning:",
                    )

            with st.container(border=True):  # --- Scaling Numerical Features ---
                st.markdown("#### Scaling")

                if st.session_state.is_data_scaled == False:
                    num_cols = (
                        st.session_state["current_df"]
                        .select_dtypes(include=["int64", "float64"])
                        .columns.tolist()
                    )
                    if num_cols:
                        scaling_method = st.radio(
                            "Select a scaling method",
                            [
                                "StandardScaler (Z-score)",
                                "MinMaxScaler (0-1 range)",
                                "Robust Scaler",
                            ],
                            horizontal=True,
                        )

                        if st.button("Apply", key="apply_scale_columns"):
                            params = {
                                "scaling_method": scaling_method,
                                "num_cols": num_cols,
                            }
                            add_operation("scale_columns", params)

                    else:
                        st.warning(
                            "No numeric columns to scale", icon=":material/warning:"
                        )
                else:
                    st.info("The DataFrame is already scaled", icon=":material/info:")

            with st.container(border=True):  # --- Encoding Categorical Variables ---
                st.markdown("#### Categorical Variables Encoding")
                if st.session_state["current_df"].isnull().sum().sum() <= 0:
                    cat_cols = (
                        st.session_state["current_df"]
                        .select_dtypes(include=["object"])
                        .columns.tolist()
                    )

                    if cat_cols:
                        encoding_method = st.radio(
                            "Select an encoding method",
                            [
                                "Label Encoding",
                                "Ordinal Encoding",
                                "One-Hot Encoding",
                                "Target Encoding",
                            ],
                            horizontal=True,
                        )

                        if encoding_method == "Target Encoding":
                            num_cols = (
                                st.session_state["current_df"]
                                .select_dtypes(include=["int64", "float64"])
                                .columns.tolist()
                            )
                            if num_cols:
                                target_column = st.selectbox(
                                    "Select a numerical column to encode against",
                                    num_cols,
                                )

                                if st.button("Apply", key="apply_encode_variables"):
                                    params = {
                                        "encoding_method": encoding_method,
                                        "cat_cols": cat_cols,
                                        "target_column": target_column,
                                    }
                                    add_operation("encode_columns", params)
                        else:
                            num_cols = (
                                st.session_state["current_df"]
                                .select_dtypes(include=["int64", "float64"])
                                .columns.tolist()
                            )

                            if st.button("Apply", key="apply_encode_variables"):
                                params = {
                                    "encoding_method": encoding_method,
                                    "num_cols": num_cols,
                                    "cat_cols": cat_cols,
                                }
                                add_operation("encode_columns", params)
                    else:
                        # st.warning("The loaded DataFrame has one or more non-numerical columns. All columns must be numerical to proceed.", icon=":material/warning:")
                        st.info(
                            "The DataFrame does not contain any categorical columns.",
                            icon=":material/info:",
                        )
                else:
                    st.warning(
                        "This operation is not possible if there are **missing values** in the DataFrame",
                        icon=":material/warning:",
                    )

        with tab6:  # --- Dimensionality Reduction ---
            st.subheader("📉 Dimensionality Reduction")
            with st.container(border=True):  # --- Variance Threshold ---
                st.markdown("#### Variance Threshold")

                num_cols = (
                    st.session_state["current_df"]
                    .select_dtypes(include=["int64", "float64"])
                    .columns.tolist()
                )
                non_num_cols = (
                    st.session_state["current_df"]
                    .select_dtypes(exclude=["number"])
                    .columns
                )

                if num_cols:
                    threshold = st.radio(
                        "Select a threshold value",
                        ["0.00", "0.01", "0.5"],
                        horizontal=True,
                    )
                    if st.button("Apply", key="apply_variance_threshold"):
                        params = {
                            "threshold": threshold,
                            "num_cols": num_cols,
                            "non_num_cols": non_num_cols,
                        }
                        add_operation("variance_threshold", params)
                else:
                    st.warning(
                        "This operation is not possible if there are no numerical columns in the DataFrame",
                        icon=":material/warning:",
                    )

            with st.container(
                border=True
            ):  # ---  Principal Component Analysis (PCA) ---
                st.markdown("####  Principal Component Analysis")

                if st.session_state.pca_done == False:
                    data = st.session_state["current_df"]
                    is_df_numeric = all_columns_numeric(data)
                    if is_df_numeric == True:
                        cols_qty = data.shape[1]
                        x_array = data.to_numpy()

                        if st.button("Verify"):
                            st.session_state.verify_clicked = True

                        if st.session_state.verify_clicked:
                            is_data_scaled, scaler = look_for_scaler()

                            if is_data_scaled is False and scaler is None:
                                st.info(
                                    "No previous scaling found, select a number of components and a scaling method to proceed",
                                    icon=":material/info:",
                                )
                                st.info(
                                    "NOTE: This operation will remove any **encoding** if undone",
                                    icon=":material/info:",
                                )
                                n_components = st.slider(
                                    "Select a number of components:", 1, cols_qty, 1, 1
                                )
                                selected_scaler = st.radio(
                                    "Select the Scaling Method",
                                    [
                                        "StandardScaler (Z-score)",
                                        "MinMaxScaler (0-1 range)",
                                        "Robust Scaler",
                                    ],
                                    key="pca_scaler",
                                    horizontal=True,
                                )
                                if selected_scaler and st.button(
                                    "Apply", key="pca_not_scaled"
                                ):
                                    params = {
                                        "selected_scaler": selected_scaler,
                                        "is_data_scaled": is_data_scaled,
                                        "x_array": x_array,
                                        "n_components": n_components,
                                    }
                                    add_operation("pca", params)

                            elif is_data_scaled and scaler:
                                st.info(
                                    f'The data was previously scaled using {scaler}, select a number of components and click the "apply" button to proceed.',
                                    icon=":material/info:",
                                )
                                # st.info("NOTE: This operation will remove any **encoding** if undone", icon=":material/info:")
                                n_components = st.slider(
                                    "Select a number of components:", 1, cols_qty, 1, 1
                                )
                                if st.button("Apply", key="pca_scaled"):
                                    params = {
                                        "is_data_scaled": is_data_scaled,
                                        "x_array": x_array,
                                        "n_components": n_components,
                                    }
                                    add_operation("pca", params)

                    else:
                        st.warning(
                            "The loaded DataFrame has one or more **non-numerical** columns. All columns must be numerical to proceed.",
                            icon=":material/warning:",
                        )

                else:
                    st.info(
                        "Principal Component Analysis already done",
                        icon=":material/info:",
                    )

        with tab7:  # --- Feature Engineering ---
            st.subheader("🎯 Feature Engineering")

            with st.container(border=True):
                st.write("To be developed...")

            # with st.container(border=True): # --- Feature Selection ---
            #     st.markdown("#### Feature Selection")
            #     target_col = st.selectbox("Select Target Column (y)", st.session_state['current_df'].columns)
            #     k = st.slider("Select number of top features", 1, len(st.session_state['current_df'].columns) - 1, 5)

            #     if st.button("Apply", key="apply_feature_selection"):
            #         if target_col:
            #             params = {'target_col': target_col,
            #                     'k': k}
            #             add_operation("feature_selection", params)
            #             st.success(f"Applied 'Feature Selection': {target_col}")
            #         else:
            #             st.warning("Please select columns.")

            # # Feature Selection (original version)
            # st.subheader("Feature Selection")
            # if st.checkbox("Select Top K Features (for supervised tasks)"):
            #     target_col = st.selectbox("Select Target Column (y)", df.columns)
            #     k = st.slider("Select number of top features", 1, len(df.columns) - 1, 5)
            #     X = df.drop(columns=[target_col])
            #     y = df[target_col]
            #     selector = SelectKBest(score_func=f_classif, k=k)
            #     X_new = selector.fit_transform(X, y)
            #     selected_features = X.columns[selector.get_support()]
            #     df = pd.concat([df[selected_features], y], axis=1)
            #     st.success(f"Selected Top {k} Features: {selected_features.tolist()}")

        with tab8:  # --- Binarization ---
            st.subheader("🧮 Binarization")
            with st.container(border=True):
                st.write("To be developed...")

        with tab9:  # --- Preprocessed Data Export ---

            if st.session_state.is_data_split:
                st.subheader("⬇️ Export Train and Test Data")
            elif (
                st.session_state.is_data_split
                and st.session_state["validation_df"] is not None
            ):
                st.subheader("⬇️ Export Train, Validation & Test Data")
            else:
                st.subheader("📄 Export Preprocessed Dataset")

            if st.session_state.is_data_split:
                with st.container(border=True):  # --- Export Data (Split) ---
                    col1, col2, col3 = st.columns([1, 6, 1])

                    with col2:
                        split_file_name = st.text_input(
                            "Enter base file name (without extension)",
                            st.session_state.get(
                                "split_output_filename_key", "processed_data"
                            ),  # Default value
                            key="split_output_filename_key",  # Unique key for this input
                        ).strip()  # Use strip to remove leading/trailing whitespace

                        if split_file_name:
                            # Download Train Data
                            if (
                                "current_df" in st.session_state
                                and st.session_state["current_df"] is not None
                                and not st.session_state["current_df"].empty
                            ):
                                csv_data_train = (
                                    st.session_state["current_df"]
                                    .to_csv(index=False)
                                    .encode("utf-8")
                                )  # Encode to bytes
                                st.download_button(
                                    label=f"Download Train Data ({split_file_name}_train.csv)",
                                    data=csv_data_train,
                                    file_name=f"{split_file_name}_train.csv",  # Add "_train" suffix
                                    mime="text/csv",
                                    use_container_width=True,
                                    icon=":material/download:",
                                )
                                # Optional: Add a success message after the button appears
                                # st.success(f"Train data available for download as {split_file_name}_train.csv") # Less intrusive

                            else:
                                st.warning(
                                    "Train data ('current_df') not available or is empty."
                                )

                            # Download Validation Data
                            if (
                                "validation_df" in st.session_state
                                and st.session_state["validation_df"] is not None
                                and not st.session_state["validation_df"].empty
                            ):
                                csv_data_validation = (
                                    st.session_state["validation_df"]
                                    .to_csv(index=False)
                                    .encode("utf-8")
                                )  # Encode to bytes
                                st.download_button(
                                    label=f"Download Validation Data ({split_file_name}_validation.csv)",
                                    data=csv_data_validation,
                                    file_name=f"{split_file_name}_validation.csv",  # Add "_validation" suffix
                                    mime="text/csv",
                                    use_container_width=True,
                                    icon=":material/download:",
                                    key="val_download_button",
                                )
                                # Optional: Add a success message after the button appears
                                # st.success(f"Test data available for download as {split_file_name}_test.csv") # Less intrusive

                            else:
                                pass
                                # st.info("Validation data ('validation_df') not available or is empty.") # Use info or warning depending on expected state

                            # Download Test Data
                            if (
                                "test_df" in st.session_state
                                and st.session_state["test_df"] is not None
                                and not st.session_state["test_df"].empty
                            ):
                                csv_data_test = (
                                    st.session_state["test_df"]
                                    .to_csv(index=False)
                                    .encode("utf-8")
                                )  # Encode to bytes
                                st.download_button(
                                    label=f"Download Test Data ({split_file_name}_test.csv)",
                                    data=csv_data_test,
                                    file_name=f"{split_file_name}_test.csv",  # Add "_test" suffix
                                    mime="text/csv",
                                    use_container_width=True,
                                    icon=":material/download:",
                                )
                                # Optional: Add a success message after the button appears
                                # st.success(f"Test data available for download as {split_file_name}_test.csv") # Less intrusive

                            else:
                                st.info(
                                    "Test data ('test_df') not available or is empty."
                                )  # Use info or warning depending on expected state

                        else:
                            # Show disabled buttons if no filename is entered
                            st.button(
                                f"Download Train Data (Enter filename)",
                                disabled=True,
                                use_container_width=True,
                                icon=":material/download:",
                            )
                            st.button(
                                f"Download Test Data (Enter filename)",
                                disabled=True,
                                use_container_width=True,
                                icon=":material/download:",
                            )

                            if st.session_state["validation_df"] is not None:
                                st.button(
                                    f"Download Validation Data (Enter filename)",
                                    disabled=True,
                                    use_container_width=True,
                                    icon=":material/download:",
                                )
                            st.caption(
                                "Please enter a base filename above to enable download buttons."
                            )

            else:
                with st.container(border=True):  # --- Export Data (NO Split)
                    col1, col2, col3 = st.columns([1, 6, 1])

                    with col2:

                        @st.cache_data  # Keep cache decorator
                        def convert_df_to_csv(df):
                            return df.to_csv(index=False).encode(
                                "utf-8"
                            )  # Already encoded

                        # Check if data exists before trying to convert and provide download button
                        if (
                            "current_df" in st.session_state
                            and st.session_state["current_df"] is not None
                            and not st.session_state["current_df"].empty
                        ):
                            csv_data = convert_df_to_csv(st.session_state["current_df"])

                            # Text input for the single output filename
                            base_filename = st.text_input(
                                "Name your output file (e.g., preprocessed_data)",
                                st.session_state.get(
                                    "output_filename_key", ""
                                ),  # Keep original key/logic
                                key="output_filename_key",
                                placeholder="Enter filename...",
                            ).strip()  # Use strip

                            # Construct the final filename and display button only if a filename is provided
                            if base_filename:
                                # Ensure the filename ends with .csv
                                if not base_filename.lower().endswith(".csv"):
                                    output_file_name = f"{base_filename}.csv"
                                else:
                                    output_file_name = base_filename

                                download_button = st.download_button(
                                    label=f"Download Full Preprocessed CSV ({output_file_name})",
                                    data=csv_data,
                                    file_name=output_file_name,  # Use the constructed filename
                                    mime="text/csv",
                                    icon=":material/download:",
                                    use_container_width=True,  # Add this for consistency
                                )
                                if download_button:
                                    # Note: st.download_button triggers client-side download.
                                    # This message appears immediately after the button is clicked, not necessarily after download completion.
                                    st.success(
                                        f"Download initiated for {output_file_name}!",
                                        icon=":material/check_circle:",
                                    )

                            else:
                                # Show disabled button if no filename is entered
                                st.button(
                                    "Download Preprocessed CSV",
                                    disabled=True,
                                    use_container_width=True,
                                    icon=":material/download:",
                                )
                                st.caption(
                                    "Please enter a filename above to enable download."
                                )

                        else:
                            st.warning(
                                "Preprocessed data ('current_df') not available or is empty for download."
                            )

        with tab10:  # --- Operations History ---
            st.subheader("📝 Applied Operations History (Log)")
            with st.container(border=True):
                if not st.session_state["operations_history"]:
                    st.caption("No operations applied yet.")
                else:
                    # Display history in reverse chronological order (most recent first)
                    for i, op in enumerate(
                        reversed(st.session_state["operations_history"])
                    ):
                        op_number = len(st.session_state["operations_history"]) - i
                        st.markdown(f"**{op_number}. Type:** `{op['op_type']}`")
                        st.markdown(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;**Params:** `{op['params']}`"
                        )
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**ID:** `{op['id']}`")
                        if i < len(st.session_state["operations_history"]) - 1:
                            st.markdown("---")

    elif uploaded_file is None:
        st.info("Please upload a CSV file using the sidebar.", icon=":material/upload:")


if __name__ == "__main__":
    st.set_page_config(page_title="Simple Preprocessing Tool", layout="wide")
    main()
