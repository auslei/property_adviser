from model_training.timeseries import train_timeseries_model


def train_models(config: Optional[Dict[str, Any]] = None):
    """
    Main function to train models - can be configured for different approaches
    """
    # Default to the new timeseries approach
    return train_timeseries_model(config)

if __name__ == "__main__":
    outcome = train_models()
    print(f"Best model: {outcome['best_model']} (saved to {outcome['model_path']})")
