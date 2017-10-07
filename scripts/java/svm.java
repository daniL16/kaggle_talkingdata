
import java.io.File;
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SMOreg;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author dani
 */
public class svm {
             public static void main(String[] args) throws IOException, Exception{
    ArffLoader reader = new ArffLoader();
    reader.setSource(new File("./train.arff"));
    Instances data = reader.getDataSet();
    Instances train_data = data.trainCV(5, 0);
    Instances test_data = data.testCV(5, 0);
    
    train_data.setClassIndex(train_data.numAttributes() - 1);
    test_data.setClassIndex(test_data.numAttributes() - 1);
   
    Classifier svm = new SMOreg();
    svm.buildClassifier(train_data);
    Evaluation ev = new Evaluation(test_data);
    ev.evaluateModel(svm,test_data);
    System.out.println("SVM  "+ev.rootMeanSquaredError());
    } 
}
