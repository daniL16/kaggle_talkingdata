/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author dani
 */
import java.io.File;
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
public class knn {
    
    public static void main(String[] args) throws IOException, Exception{
    ArffLoader reader = new ArffLoader();
    reader.setSource(new File("./train.arff"));
    Instances data = reader.getDataSet();
    Instances train_data = data.trainCV(5, 0);
    Instances test_data = data.testCV(5, 0);
    
    train_data.setClassIndex(train_data.numAttributes() - 1);
    test_data.setClassIndex(test_data.numAttributes() - 1);
    Classifier ibk = new IBk();		
    ibk.buildClassifier(train_data);

    Evaluation ev = new Evaluation(test_data);
    ev.evaluateModel(ibk,test_data);
    System.out.println("KNN "+ev.rootMeanSquaredError());
    }
}
