package com.learning.trend.service;

import com.learning.trend.model.FarmerInputFeatures;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.impl.ArrayExample;

import java.io.FileInputStream;
import java.io.ObjectInputStream;

@Service
public class TribuoPredictionService {

    private static final Logger LOG = LoggerFactory.getLogger(TribuoPredictionService.class);
    public static final String MODEL_PATH = "src/main/resources/model/farmer_input_timeseries.ser";

    private Model<Label> model;

    @PostConstruct
    public void loadModel() throws Exception {

        LOG.info("Loading Tribuo model from path: {}", MODEL_PATH);

        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream(MODEL_PATH))) {
            model = (Model<Label>) ois.readObject();
        }
    }

    public String predict(FarmerInputFeatures features) {
        Example<Label> example = buildExample(features);

        Prediction<Label> prediction = model.predict(example);

        return prediction.getOutput().getLabel();
    }

    public Example<Label> buildExample(FarmerInputFeatures f) {

        ArrayExample<Label> example = new ArrayExample<>(new Label("previous_input_usage"));

        example.add(new Feature("year", f.getYear()));
        example.add(new Feature("month", f.getMonth()));

        example.add(new Feature("crop_type@" + f.getCropType(), 1.0));
        example.add(new Feature("soil_type@" + f.getSoilType(), 1.0));
        example.add(new Feature("region@" + f.getRegion(), 1.0));
        example.add(new Feature("season@" + f.getSeason(), 1.0));

        example.add(new Feature("rainfall", f.getRainfall()));
        example.add(new Feature("farmer_land_size", f.getFarmerLandSize()));

        return example;
    }
}