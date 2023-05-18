package com.example.hacking;

import ch.qos.logback.core.subst.Tokenizer;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;


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


    public DataSet prepareDataSet(String labelsFilePath, String textFilePath) throws IOException {
        // Czytanie plików JSON
        String labelsContent = new String(Files.readAllBytes(Paths.get(labelsFilePath)));
        String textContent = new String(Files.readAllBytes(Paths.get(textFilePath)));

        // Parsowanie JSON
        JSONObject labelsJson = new JSONObject(labelsContent);
        JSONObject textJson = new JSONObject(textContent);

        List<String> labelsList = new ArrayList<>();
        List<String> featuresList = new ArrayList<>();

        for (String key : textJson.keySet()) {
            String text = textJson.getString(key);
            String label = labelsJson.getString(textJson.getString(key));
            labelsList.add(label);
            featuresList.add(text);
        }

        int numClasses = labelsJson.keySet().size();

        // Teraz musimy przekształcić te listy w odpowiedni format dla DataSet
        // Zakładam, że tekst jest już wektorowo zapisany. Jeśli nie, musisz go przekształcić, np. za pomocą TF-IDF lub Word2Vec.

        int numExamples = labelsList.size();
        int vectorSize = featuresList.get(0).length();

        float[][] featureMatrix = new float[numExamples][vectorSize];
        float[][] labelMatrix = new float[numExamples][numClasses];

        for (int i = 0; i < numExamples; i++) {
            String features = featuresList.get(i);
            String label = labelsList.get(i);

            for (int j = 0; j < vectorSize; j++) {
                featureMatrix[i][j] = (float) features.charAt(j);
            }

            int labelIndex = Integer.parseInt(label);
            labelMatrix[i][labelIndex] = 1.0f;
        }

        INDArray featuresNDArray = Nd4j.create(featureMatrix);
        INDArray labelsNDArray = Nd4j.create(labelMatrix);

        return new DataSet(featuresNDArray, labelsNDArray);
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


}
