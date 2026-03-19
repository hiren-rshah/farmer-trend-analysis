package com.learning.trend.controller;

import com.learning.trend.model.FarmerInputDTO;
import com.learning.trend.service.ChatService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/trend")
public class ChatController {

    private static final Logger LOG = LoggerFactory.getLogger(ChatController.class);
    @Autowired
    private ChatService chatService;

    @PostMapping (path = "/chat")
    public String chat(@RequestBody String userInput)
    {
        LOG.info("Received user input: {}", userInput);
        return chatService.processUsingLLM(userInput);
    }

    @PostMapping (path = "/api", consumes = "application/json")
    public String api(@RequestBody FarmerInputDTO farmerInputDTO)
    {
        LOG.info("Received API input: {}", farmerInputDTO.toString());
        return chatService.process(farmerInputDTO);
    }

}