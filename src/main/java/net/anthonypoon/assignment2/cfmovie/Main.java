/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package net.anthonypoon.assignment2.cfmovie;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.Random;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

/**
 *
 * @author ypoon
 */
public class Main {
    private static double hyperU = 0.2;
    private static double hyperV = 0.2;
    private static RealMatrix identityMatrix;
    private static RealMatrix trainMatrix;
    private static int k = 1;
    private static int n = 544;
    private static int m = 1682;
    public static void main(String[] args) throws Exception {
        Options options = new Options();
        options.addOption("t", "train", true, "Training set");
        options.addOption("T", "test", true, "testing set");
        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);
        BufferedReader br = new BufferedReader(new FileReader(cmd.getOptionValue("train")));
        String line;
        trainMatrix = new Array2DRowRealMatrix(n, m);
        identityMatrix = new Array2DRowRealMatrix(n, m);
        while ((line = br.readLine()) != null) {
            String[] strArray = line.split("\t");
            trainMatrix.addToEntry(Integer.valueOf(strArray[0]) - 1, Integer.valueOf(strArray[1]) - 1, Integer.valueOf(strArray[2]));
            identityMatrix.addToEntry(Integer.valueOf(strArray[0]) - 1, Integer.valueOf(strArray[1]) - 1, 1);
        }
        br.close();
        Random random = new Random() ;
        RealMatrix uMatrix = new Array2DRowRealMatrix(n, k);
        RealMatrix vMatrix = new Array2DRowRealMatrix(k, m);
        for (int i = 0; i < n; i ++) {
            for (int j = 0; j < k; j ++) {
                uMatrix.setEntry(i, j, random.nextInt(5) + 1);
            }
        }
        for (int i = 0; i < k; i ++) {
            for (int j = 0; j < n; j ++) {
                vMatrix.setEntry(i, j, random.nextInt(5) + 1);
            }
        }
        RealMatrix result = getPrediction(uMatrix, vMatrix);
        br = new BufferedReader(new FileReader(cmd.getOptionValue("test")));
        PrintWriter writer = new PrintWriter("E.test.rs.txt");
        while ((line = br.readLine()) != null) {
            String[] strArray = line.split("\t");
            writer.println(String.format("%.3f", result.getEntry(Integer.valueOf(strArray[0]) - 1, Integer.valueOf(strArray[1])-1)));
        }
    }
    
    private static RealMatrix getPrediction(RealMatrix uMatrix, RealMatrix vMatrix) {
        double stepSize = 0.01;
        double error = 0.0;
        double lastError = 0.0;
        RealMatrix lastRunU;
        RealMatrix lastRunV;
        do {
            lastError = error;
            lastRunU = uMatrix.copy();
            lastRunV = vMatrix.copy();
            for (int i = 0; i < n; i ++) {
                for (int j = 0; j < m; j++) {
                    for (int x = 0; x < k; x++) {
                        if (trainMatrix.getEntry(i, j) > 0) {
                            double gradientU = (uMatrix.getEntry(i, x) * vMatrix.getEntry(x, j) - trainMatrix.getEntry(i, j)) * vMatrix.getEntry(x, j) + hyperU * uMatrix.getEntry(i, x);
                            double gradientV = (uMatrix.getEntry(i, x) * vMatrix.getEntry(x, j) - trainMatrix.getEntry(i, j)) * uMatrix.getEntry(i, x) + hyperV * vMatrix.getEntry(x, k);
                            uMatrix.setEntry(i, x,  uMatrix.getEntry(i, x) - stepSize * gradientU);
                            vMatrix.setEntry(x, j,  vMatrix.getEntry(x, j) - stepSize * gradientV);
                        }
                    }                
                }
            }
            error = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    if (trainMatrix.getEntry(i, j) > 0) {
                        error = Math.pow(trainMatrix.getEntry(i, j) - uMatrix.getRowVector(i).dotProduct(vMatrix.getColumnVector(j)), 2);
                        for (int x = 0; x < k; x ++) {
                            error = error + hyperU / 2 * Math.pow(uMatrix.getEntry(i, x), 2) + hyperV / 2 * Math.pow(vMatrix.getEntry(x, k), 2);
                        }
                    }
                }
            }
            System.out.println(error);
        } while (lastError == 0 || error < lastError);
        uMatrix = lastRunU.copy();
        vMatrix = lastRunV.copy();
        return uMatrix.multiply(vMatrix);
    }
}
