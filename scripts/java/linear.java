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
import weka.classifiers.functions.GaussianProcesses;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.functions.LinearRegression;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author dani
 */
public class linear {
    public static void main(String[] args) throws IOException, Exception{
    ArffLoader reader = new ArffLoader();
    reader.setSource(new File("./train.arff"));
    Instances data = reader.getDataSet();
    Instances train_data = data.trainCV(5, 0);
    Instances test_data = data.testCV(5, 0);
    
    train_data.setClassIndex(train_data.numAttributes() - 1);
    test_data.setClassIndex(test_data.numAttributes() - 1);
   
    Classifier rf = new LinearRegression();
    rf.buildClassifier(train_data);
    Evaluation ev = new Evaluation(test_data);
    ev.evaluateModel(rf,test_data);
    System.out.println("Gaussian  "+ev.rootMeanSquaredError());
    }
}

