package com.example.hacking;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;

@SpringBootApplication
@Slf4j
public class HackingApplication {

    public static void main(String[] args) throws IOException {
        SpringApplication.run(HackingApplication.class, args);
        var jsonToDataSet = new JsonToDataSet();
        String trainingLabelsFilePath = "C:\\Users\\Adam\\IdeaProjects\\hacking\\src\\main\\resources\\DataSet\\train_labels_final.json";
        String trainingSetFilePath = "C:\\Users\\Adam\\IdeaProjects\\hacking\\src\\main\\resources\\DataSet\\train_set_ocr.json";
//        jsonToDataSet.jsonToDataSet();
        // After the model is trained, you can use it to convert words into vectors.
        // For example, to get the vector for the word "example", you would use:
//

        DataSet dataSet = jsonToDataSet.prepareDataSet(trainingLabelsFilePath, trainingSetFilePath);

    }

}
