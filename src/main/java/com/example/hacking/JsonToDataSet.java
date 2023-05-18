package com.example.hacking;


import lombok.extern.slf4j.Slf4j;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.json.JSONException;
import org.json.JSONObject;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Slf4j
public class JsonToDataSet {

    // Odczyt danych z JSON i wczytanie do DataSet
    public DataSet jsonToDataSet(String jsonString) throws JSONException, IOException, InterruptedException {
        // Tworzenie JSONObject z ciągu znaków JSON
        JSONObject jo = new JSONObject(jsonString);

        // Tworzenie RecordReader
        RecordReader recordReader = new CSVRecordReader(0, ',');
        recordReader.initialize(new FileSplit(new ClassPathResource(jo.getString("filename")).getFile()));

        // Tworzenie DataSetIterator
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, jo.getInt("numLines"), jo.getInt("featuresCount"), jo.getInt("classesCount"));
        DataSet allData = iterator.next();

        // Mieszanie danych
        allData.shuffle(jo.getInt("seed"));

        // Normalizacja danych
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);

        // Podział danych na zestawy treningowy i testowy
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(jo.getDouble("trainingFraction"));
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        // Zwrócenie zestawu danych
        return trainingData;
    }




    private static Word2Vec createWorld2Vec(String filePath) throws FileNotFoundException {
        SentenceIterator iter = new BasicLineIterator(filePath);

        // Create a tokenizer to split sentences into individual words.
        TokenizerFactory t = new DefaultTokenizerFactory();

        // Train the Word2Vec model.
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(1)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();
        // Check if the vocabulary is empty.
        if (vec.getVocab().numWords() == 0) {
            log.error("The vocabulary is empty. Check the input data and the tokenizer.");
        }
        vec.fit();
        return vec;
    }

    public void trainWord2Vec(List<String> sentences) {
        // Krok 1: Przygotowanie danych do treningu Word2Vec
        SentenceIterator iter = new CollectionSentenceIterator(sentences);

        // Krok 2: Tworzenie modelu Word2Vec
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5) // Minimalna częstotliwość słowa, które ma być uwzględnione w słowniku
                .iterations(1)
                .layerSize(100) // Rozmiar wektorów
                .seed(42)
                .windowSize(5) // Rozmiar okna kontekstu
                .iterate(iter) // Iterator po zdaniach
                .tokenizerFactory(new DefaultTokenizerFactory()) // Tokenizer do dzielenia zdań na słowa
                .build();

        // Krok 3: Trenowanie modelu Word2Vec
        vec.fit();
    }

    public INDArray transformTextWithWord2Vec(Word2Vec vec, String text) {
        // Przekształcanie tekstu na wektor przy użyciu Word2Vec
        TokenizerFactory t = new DefaultTokenizerFactory();
        Tokenizer tokenizer = t.create(text);
        List<String> tokens = tokenizer.getTokens();
        INDArray features = Nd4j.create(1, vec.getLayerSize());

        for (String token : tokens) {
            INDArray wordVector = vec.getWordVectorMatrix(token);
            if (wordVector != null) {
                features.addi(wordVector);
            }
        }

        features.divi(tokens.size());
        return features;
    }




    public DataSet prepareDataSet(String labelsFilePath, String textFilePath) throws IOException {
        // Reading JSON files
        String labelsContent = new String(Files.readAllBytes(Paths.get(labelsFilePath)));
        String textContent = new String(Files.readAllBytes(Paths.get(textFilePath)));

        // Parsing JSON
        JSONObject labelsJson = new JSONObject(labelsContent);
        JSONObject textJson = new JSONObject(textContent);
       var w2VecLabels= createWorld2Vec(labelsFilePath);
       var w2VecWorld= createWorld2Vec(textFilePath);
        // Prepare sentences for Word2Vec training
        List<String> sentences = new ArrayList<>();
        for (String key : textJson.keySet()) {
            String text = textJson.getString(key);
            sentences.addAll(Arrays.asList(text.split(" ")));
        }
        // Train Word2Vec model on all sentences
        trainWord2Vec(sentences);

        List<DataSet> allData = new ArrayList<>();
        for (String key : textJson.keySet()) {
            String text = textJson.getString(key);
            String label = labelsJson.getString(key);

            // Vectorize text using Word2Vec
            INDArray featuresRow = transformTextWithWord2Vec(w2VecWorld, text);

            INDArray labelsRow = Nd4j.create(1, labelsJson.keySet().size());
            labelsRow.putScalar(labelsJson.getInt(label), 1.0);

            allData.add(new DataSet(featuresRow, labelsRow));
        }

        int batchSize = 50;
        ListDataSetIterator<DataSet> iterator = new ListDataSetIterator<>(allData, batchSize);

        List<INDArray> features = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();
        while (iterator.hasNext()) {
            DataSet dataSet = iterator.next();
            features.add(dataSet.getFeatures());
            labels.add(dataSet.getLabels());
        }

        // If you don't use masking, pass null for the mask arguments
        return new DataSet(Nd4j.vstack(features), Nd4j.vstack(labels), null, null);
    }


}
