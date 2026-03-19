package com.learning.trend.ml;

import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.impl.ArrayExample;

import java.io.FileInputStream;
import java.io.ObjectInputStream;


public class FarmerInputPredictor {
    public static final String MODEL_PATH = "src/main/resources/model/farmer_input_timeseries.ser";

    public static Model<?> loadModel(String path) throws Exception {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            return (Model<?>) ois.readObject();
        }
    }

    public static void predict(Model<Label> model) {

        Example<Label> example = new ArrayExample<>(new Label("previous_input_usage"));

        // Numeric features
        example.add(new Feature("year", 2023.0));
        example.add(new Feature("month", 7));
        example.add(new Feature("rainfall_mm", 100));
        example.add(new Feature("farmer_land_size_acres", 1.5));

        // Categorical (One-hot encoding)
        example.add(new Feature("crop_type@Wheat", 1.0));
        example.add(new Feature("soil_type@Black", 1.0));
        example.add(new Feature("region@Maharashtra", 1.0));
        example.add(new Feature("season@Kharif", 1.0));

        // Predict
        model.getFeatureIDMap().forEach(info -> System.out.println("Model FeatureID Map:"+ info.getName()));
        Prediction<Label> prediction = model.predict(example);

        System.out.println("Predicted Input Usage: " + prediction.getOutput());
    }


    public static void main(String[] args) throws Exception {


        // 1. Load model
        Model<Label> loadedModel = (Model<Label>) loadModel(MODEL_PATH);

        // 2. Predict
        predict(loadedModel);
    }

}