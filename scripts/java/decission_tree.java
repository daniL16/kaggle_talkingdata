
import java.io.File;
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.REPTree;
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
public class decission_tree {
    
         public static void main(String[] args) throws IOException, Exception{
    ArffLoader reader = new ArffLoader();
    reader.setSource(new File("./train.arff"));
    Instances data = reader.getDataSet();
    Instances train_data = data.trainCV(5, 0);
    Instances test_data = data.testCV(5, 0);
    
    train_data.setClassIndex(train_data.numAttributes() - 1);
    test_data.setClassIndex(test_data.numAttributes() - 1);
    Classifier lm = new REPTree();
    lm.buildClassifier(train_data);
    Evaluation ev = new Evaluation(test_data);
    ev.evaluateModel(lm,test_data);
    System.out.println("Decision Tree  "+ev.rootMeanSquaredError());
    }
}
