package com.learning.trend.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.learning.trend.model.FarmerInputDTO;
import com.learning.trend.model.FarmerInputFeatures;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.stereotype.Service;

@Service
public class ChatService {

    private static final Logger LOG = LoggerFactory.getLogger(ChatService.class);


    private final TribuoPredictionService predictionService;

    private final ChatModel chatModel;


    private final ObjectMapper objectMapper;

    public ChatService(TribuoPredictionService predictionService,
                       ChatModel chatModel,
                       ObjectMapper objectMapper) {
        this.predictionService = predictionService;
        this.chatModel = chatModel;
        this.objectMapper = objectMapper;
    }

    /**
     * User Chat Prompt -> LLM Extraction → JSON → DTO → Feature Mapping → ML Prediction → LLM Explanation
     * @param userInput
     * @return
     */
    public String processUsingLLM(String userInput) {

        // Step 1: LLM → JSON string
        // Create Prompt
        Prompt prompt = new Prompt("""
                    Return ONLY a valid JSON object.
                    Do NOT include markdown, backticks, or explanation.
                   
                    STRICT REQUIREMENTS:
                    - Include ALL fields: year, month, crop_type, soil_type, region, season, rainfall, farmer_land_size, previous_input_usage
                    - If any value is missing, set it to null
                    - Numeric fields must be numbers
                    - Return only 1 JSON object, do NOT return an array
                   
                    Output format:
                    {
                     "year": number,
                     "month": number,
                     "crop_type": string,
                     "soil_type": string,
                     "region": string,
                     "season": string,
                     "rainfall": number,
                     "farmer_land_size": number,
                     "previous_input_usage": string or null
                    }
                    Input: %s
                """.formatted(userInput));

        LOG.info("Sending this prompt to LLM for JSON extraction: \n{}", prompt.getContents());

        LOG.info("chatModel used is: "+ chatModel.getClass().getName());
        String structuredJson = chatModel.call(prompt)
                .getResult()
                .getOutput()
                .getText();

        // Step 2: JSON → DTO (Entity Mapping)
        LOG.info("Received structured JSON from LLM: \n{}", structuredJson);

        FarmerInputDTO dto = mapToDTO(structuredJson);

        // Step 3: DTO → Features
        FarmerInputFeatures features = mapToFeatures(dto);

        // Step 4: ML Prediction
        LOG.info("Performing ML prediction with features: {}", features.toString());

        String prediction = predictionService.predict(features);

        LOG.info("ML Prediction result: {}", prediction);

        // Step 5: Response formatting
        LOG.info("Sending this prompt to LLM for explanation in farmer-friendly language");
        String explanation = chatModel.call(
                new Prompt("""
                    Explain this prediction in simple farmer-friendly language:
                    %s
                """.formatted(prediction))
        ).getResult().getOutput().getText();

        LOG.info("LLM Explanation: \n{}", explanation);

        String return_text ="Prediction: %s\nExplanation: %s".formatted(prediction, explanation);
        return return_text;
    }


    public String process(FarmerInputDTO farmerInputDTO) {

        // Step 1: LLM → JSON string
        // Create Prompt
        // Step 2: JSON → DTO (Entity Mapping)
        // Step 1 and 2 not required as done in previous steps because This Method alrady receives structure DTO as input
        // Step 3: DTO → Features
        FarmerInputFeatures features = mapToFeatures(farmerInputDTO);

        // Step 4: ML Prediction
        LOG.info("Performing ML prediction with features: {}", features.toString());
        String prediction = predictionService.predict(features);

        LOG.info("ML Prediction result: {}", prediction);

        // Step 5: Response formatting
        // Not required because this method is expected to return prediction directly without any explanation. Explanation is only required when user input is in unstructured format and we need to extract structured data from it and then predict and explain the prediction in farmer friendly language.

        return prediction;
    }

    // 🔹 Entity Mapper (replacement for .entity())
    private FarmerInputDTO mapToDTO(String json) {
        try {
            return objectMapper.readValue(json, FarmerInputDTO.class);
        } catch (Exception e) {
            throw new RuntimeException("Failed to map LLM response to DTO. Response was: " + json, e);
        }
    }

    // 🔹 DTO → Feature mapping
    private FarmerInputFeatures mapToFeatures(FarmerInputDTO dto) {
        FarmerInputFeatures f = new FarmerInputFeatures();
        f.setYear(dto.getYear());
        f.setMonth(dto.getMonth());
        f.setCropType(dto.getCropType());
        f.setSoilType(dto.getSoilType());
        f.setRegion(dto.getRegion());
        f.setSeason(dto.getSeason());
        f.setRainfall(dto.getRainfall());
        f.setFarmerLandSize(dto.getFarmerLandSize());
//        f.setPreviousUsage(dto.getPreviousInputUsage()); // NOT Required because previous usage is expected output
        return f;
    }
}