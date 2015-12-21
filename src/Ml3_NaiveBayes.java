import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ArffLoader;


/**
 * @author vaibhav
 *
 */
public class Ml3_NaiveBayes {

	private static String posClass, negClass;
	private static int numOfNegClass = 0, numOfPosClass = 0;

	/*private static final String ARFF_FILE_PATH = 
			"src/vote_train.arff";

	private static final String ARFF_FILE_PATH_TEST = 
			"src/vote_test.arff";
*/
	private static HashMap<String, Double> hash = new HashMap <String, Double>();
	/*private static HashMap<Integer, Integer> graph = new HashMap <Integer, Integer>();
*/
	void setClassLabels (Instances instances) {
		instances.setClassIndex(instances.numAttributes()-1);
		String s = instances.attribute(instances.numAttributes()-1).toString();
		s = s.substring(s.indexOf("{"));

		negClass = s.substring(1, s.indexOf(","));
		posClass = s.substring(s.indexOf(",") + 1, s.length()-1);

		//System.out.println(negClass + " " + posClass);
	}

	int[] getParents(pair[] pairs, int len, int v2) {
		int[] parents = new int[len];
		int index = 0;

		for(int i=0; i < len; i++) {
			if(pairs[i].v2 == v2) parents[index++] = pairs[i].v1;
		}

		int[] returnParents = new int[index];
		for(int i = 0; i < index; i++) 
			returnParents[i] = parents[i];
		return returnParents;
	}

	int extractAttrInd (String s) {
		s = s.substring(0, s.indexOf("_"));
		return Integer.parseInt(s);
	}

	void printClassNames(Instances dataInstances) {
		int n = dataInstances.numAttributes();
		String s;
		for(int i = 0; i < n-1; i++) {

			s = dataInstances.attribute(i).name();
			//System.out.println(s + " " + dataInstances.attribute(n-1).name());
			System.out.println(s + " class ");
		}
		System.out.println();
	}

	String getIndexOfAttrValue(String attr, String val) {
		String[] ans = null;

		attr = attr.substring(attr.indexOf("{")+1, attr.indexOf("}"));
		//System.out.println(attr);
		ans = attr.split(",");

		for(int i = 0; i < ans.length; i++) {

			if(ans[i].equals(val)) {
				//System.out.println(i + ".0");
				return i + ".0";
			}
		}
		return null;
		//return ans;
	}


	void printAttrAndParents(pair[] pairs, int len, Instances dataInstances) {
		int n = dataInstances.numAttributes();
		String s;
		for(int i = 0; i < n-1; i++) {

			s = dataInstances.attribute(i).name();
			System.out.print(s + " ");

			int[] parents = getParents(pairs, len, i);
			for(int j = 0; j < parents.length; j++)
				System.out.print(dataInstances.attribute(parents[j]).name() + " ");
			System.out.println("class ");
		}
		System.out.println();
	}


	void tan(Instances dataInstances, Instances testInstances) {
		int count = 0;

		int n = dataInstances.numInstances();
		int m = dataInstances.numAttributes()-1;

		//System.out.println(dataInstances.instance(0).value(m));

		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {

				String hashKey = j + "_" + dataInstances.instance(i).value(j) + "_";

				if(dataInstances.instance(i).stringValue(m).equals(negClass)) 
					hashKey += negClass;
				else 
					hashKey += posClass;

				//System.out.println(hashKey);
				if(hash.containsKey(hashKey)) {
					hash.put(hashKey, hash.get(hashKey) + 1);
				} else {
					//added laplase too with count 1
					hash.put(hashKey, 2.0);
				}
			}
		}

		Iterator it = hash.entrySet().iterator();
		double temp;
		int attrInd ;

		while(it.hasNext() == true) {
			HashMap.Entry pair = (HashMap.Entry)it.next();

			if(pair.getKey().toString().contains(negClass)) {
				temp = numOfNegClass;
			} else {
				temp = numOfPosClass;
			}

			attrInd = extractAttrInd(pair.getKey().toString());
			temp += dataInstances.attribute(attrInd).numValues();

			hash.put(pair.getKey().toString(), Double.parseDouble(pair.getValue().toString())/(double)temp);
		}

		int numInst = dataInstances.numInstances();
		Double[][] values = new Double[m][m];

		for(int i = 0; i < m; i++) {
			for(int j = 0; j < m; j++) {

				if(i == j) {
					values[i][i] = -1.0;
				} else {

					values[i][j] = 0.0;

					for(int k = 0; k < 2; k++) {

						for(int x = 0; x < dataInstances.attribute(i).numValues(); x++) {
							for(int y = 0; y < dataInstances.attribute(j).numValues(); y++) {

								int count_xi_xj_y = 0;
								for(int r = 0; r < numInst; r++) {

									if(dataInstances.instance(r).value(i) == (double)x 
											&& dataInstances.instance(r).value(j) == (double)y 
											&& dataInstances.instance(r).value(m) == (double)k) {
										count_xi_xj_y ++;
									}
								}

								double p_xi_xj_y = ((double)count_xi_xj_y + 1.0) / 
										(numInst + dataInstances.attribute(i).numValues() * dataInstances.attribute(j).numValues() * 2);

								int temp2;
								if(k == 0) temp2 = numOfNegClass;
								else       temp2 = numOfPosClass;

								double p_xi_xj_given_y = ((double)count_xi_xj_y + 1.0) / 
										(temp2 + dataInstances.attribute(i).numValues() * dataInstances.attribute(j).numValues());

								String tempS1 = getIndexOfAttrValue(dataInstances.attribute(i).toString(), dataInstances.attribute(i).value(x));
								String tempS2 = getIndexOfAttrValue(dataInstances.attribute(j).toString(), dataInstances.attribute(j).value(y));

								String hashKey_xi = i + "_" + tempS1 + "_"; 
								String hashKey_xj = j + "_" + tempS2 + "_"; 

								if(k == 0) {hashKey_xi += negClass; hashKey_xj += negClass;}
								else       {hashKey_xi += posClass; hashKey_xj += posClass;}

								//System.out.println(hashKey_xj);

								double p_xi_given_y , p_xj_given_y ;

								if(!hash.containsKey(hashKey_xi)) {
									if(k == 0) { 
										hash.put(hashKey_xi, 1.0/(numOfNegClass + dataInstances.attribute(i).numValues())); 
									} else {
										hash.put(hashKey_xi, 1.0 / ( numOfPosClass + dataInstances.attribute(i).numValues() ));  
									}
								}

								if(!hash.containsKey(hashKey_xj)) {
									if(k == 0) { 
										hash.put(hashKey_xj, 1.0/(numOfNegClass + dataInstances.attribute(j).numValues())); 
									} else {
										hash.put(hashKey_xj, 1.0 / ( numOfPosClass + dataInstances.attribute(j).numValues() ));  
									}
								}

								p_xi_given_y = hash.get(hashKey_xi);
								p_xj_given_y = hash.get(hashKey_xj);

								//System.out.println("i:" + i + ", j:" + j + ",y:" + y + ", ::" + p_xi_xj_y  + " " + p_xi_xj_given_y + " " + p_xi_given_y + " " + p_xj_given_y);

								values[i][j] += p_xi_xj_y * Math.log(p_xi_xj_given_y / (p_xi_given_y * p_xj_given_y))/Math.log(2);
							}
						}
					}
				}
			}
		}

		pair[] pairs = new pair[100];
		for(int i = 0; i < 100; i++) {
			pairs[i] = new pair();
		}

		int index = 0;

		Boolean[] graphVertices = new Boolean[m];
		graphVertices[0] = true;
		for(int i = 1; i < m; i++) {
			graphVertices[i] = false;
		}

		while(true) {

			int maxRow = 0;
			int maxCol = 0;
			double maxVal = -1;

			for(int i = 0; i < m; i++) {

				if(graphVertices[i] == true) {
					for(int j = 0; j < m; j++) {
						if(values[i][j] > maxVal) {

							if(graphVertices[j] == true)
								continue;

							maxRow = i;
							maxCol = j;
							maxVal = values[i][j];
						}
					}
				}

			}

			if(maxVal != -1) {
				pairs[index].v1 = maxRow;
				pairs[index++].v2 = maxCol;
				graphVertices[maxCol] = true;
			} else {
				break;
			}
		}
		/*
		for(int i = 0; i < index; i++) {
			System.out.print(pairs[i].toString() + " ");
		}
		System.out.println("\n");

		int[] temp5= getParents(pairs, index, 7);
		for(int i = 0; i < temp5.length; i++) System.out.print(temp5[i]+" ");
		System.out.println();
		 */

		printAttrAndParents(pairs, index, dataInstances);

		int n_test = testInstances.numInstances();

		double ansPos;
		double ansNeg;

		for(int i = 0; i < n_test; i++) {

			ansPos = 1.0;
			ansNeg = 1.0;

			for(int k = 0; k < 2; k++) {

				for(int j = 0; j < m; j++) {

					int count_xj_y = 0;
					int count_xi_xj_y = 0;
					int[] parents = getParents(pairs, index, j);

					for(int r = 0; r < numInst; r++) {

						if(dataInstances.instance(r).value(j) == testInstances.instance(i).value(j)
								&& dataInstances.instance(r).value(m) == (double)k) {

							boolean flag2 = false;
							for(int s = 0; s < parents.length; s++) {
								if(dataInstances.instance(r).value(parents[s]) != testInstances.instance(i).value(parents[s])) {
									flag2 = true;
									break;
								}
							}
							if(flag2 == false) {
								count_xi_xj_y ++;
							}
						}


						if(dataInstances.instance(r).value(m) == (double)k) {
							boolean flag = false;
							for(int s = 0; s < parents.length; s++) {
								if(dataInstances.instance(r).value(parents[s]) != testInstances.instance(i).value(parents[s])) {
									flag = true;
									break;
								}
							}

							if(flag == false) {
								count_xj_y ++;
							}

						}

					}

					if(k == 0)
						ansNeg *= ((double)count_xi_xj_y +1) / (count_xj_y + dataInstances.attribute(j).numValues());
					else
						ansPos *= ((double)count_xi_xj_y +1) / (count_xj_y + dataInstances.attribute(j).numValues());
				}
			}

			ansPos *= (((double)numOfPosClass+1) /(numOfNegClass + numOfPosClass+2));
			ansNeg *= (((double)numOfNegClass+1) /(numOfNegClass + numOfPosClass+2));
			/*
			int count_root_c = 0;

			for(int r = 0; r < numInst; r++) {

				if(dataInstances.instance(r).value(m) == 0 
						&& dataInstances.instance(r).value(0) == testInstances.instance(i).value(0)) {
					count_root_c ++;
				}
			}	

			ansNeg *= ((double)count_root_c + 1) / (numOfNegClass + dataInstances.attribute(0).numValues());

			count_root_c = 0;

			for(int r = 0; r < numInst; r++) {

				if(dataInstances.instance(r).value(m) == 0 
						&& dataInstances.instance(r).value(0) == testInstances.instance(i).value(0)) {
					count_root_c ++;
				}
			}	
			ansPos *= ((double)count_root_c + 1) / (numOfPosClass + dataInstances.attribute(0).numValues());
			 */			
			double temp2 = ansPos;

			ansPos = ansPos / (ansPos + ansNeg);
			ansNeg = ansNeg / (temp2 + ansNeg);

			//System.out.println(i + ": " + ansNeg + " " + ansPos);

			if(ansNeg > ansPos) {
				System.out.println(negClass + " " + testInstances.instance(i).stringValue(m) + " " + ansNeg);

				if(testInstances.instance(i).stringValue(m).equals(negClass)) {
					count++;
				}
			} else {
				System.out.println(posClass + " " + testInstances.instance(i).stringValue(m) + " " + ansPos);
				if(testInstances.instance(i).stringValue(m).equals(posClass)) {
					count++;
				}
			}
		}

		System.out.print("\n" + count + " " + n_test + " " + (double)count/n_test);
	}

	void naiveBayes(Instances dataInstances, Instances testInstances) {

		int count = 0;
		printClassNames(dataInstances);

		int n = dataInstances.numInstances();
		int m = dataInstances.numAttributes()-1;

		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {

				String hashKey = j + "_" + dataInstances.instance(i).value(j) + "_";

				if(dataInstances.instance(i).stringValue(m).equals(negClass)) 
					hashKey += negClass;
				else 
					hashKey += posClass;

				if(hash.containsKey(hashKey)) {
					hash.put(hashKey, hash.get(hashKey) + 1);
				} else {
					//added laplase too with count 1
					hash.put(hashKey, 2.0);
				}
			}
		}

		Iterator it = hash.entrySet().iterator();
		double temp;
		int attrInd ;

		while(it.hasNext() == true) {
			HashMap.Entry pair = (HashMap.Entry)it.next();

			if(pair.getKey().toString().contains(negClass)) {
				temp = numOfNegClass;
			} else {
				temp = numOfPosClass;
			}

			attrInd = extractAttrInd(pair.getKey().toString());
			temp += dataInstances.attribute(attrInd).numValues();

			hash.put(pair.getKey().toString(), Double.parseDouble(pair.getValue().toString())/(double)temp);
		}

		int n_test = testInstances.numInstances();

		double ansPos;
		double ansNeg;

		for(int i = 0; i < n_test; i++) {

			ansPos = 1.0;
			ansNeg = 1.0;

			for(int k = 0; k < 2; k++) {

				for(int j = 0; j < m; j++) {

					String hashKey = j + "_" + testInstances.instance(i).value(j) + "_";

					String key = hashKey;

					if(k == 0) key += negClass;
					else key += posClass;

					if(hash.containsKey(key)) {
						if(k == 0)	
							ansNeg *= hash.get(key);
						else 
							ansPos *= hash.get(key);

					} else {

						if(k == 0) { 
							hash.put(key, 1.0/(numOfNegClass + dataInstances.attribute(j).numValues())); 
							ansNeg = hash.get(key) * ansNeg;
						} else {
							hash.put(key, 1.0 / ( numOfPosClass + dataInstances.attribute(j).numValues() ));  
							ansPos = hash.get(key) * ansPos;
						}
					}
				}
			}

			ansPos *= (((double)numOfPosClass+1) /(numOfNegClass + numOfPosClass+2));
			ansNeg *= (((double)numOfNegClass+1) /(numOfNegClass + numOfPosClass+2));

			double temp2 = ansPos;

			ansPos = ansPos / (ansPos + ansNeg);
			ansNeg = ansNeg / (temp2 + ansNeg);

			//System.out.println(i + ": " + ansNeg + " " + ansPos);

			if(ansNeg > ansPos) {
				System.out.println(negClass + " " + testInstances.instance(i).stringValue(m) + " " + ansNeg);

				if(testInstances.instance(i).stringValue(m).equals(negClass)) {
					count++;
				}
			} else {
				System.out.println(posClass + " " + testInstances.instance(i).stringValue(m) + " " + ansPos);
				if(testInstances.instance(i).stringValue(m).equals(posClass)) {
					count++;
				}
			}
		}

		System.out.print("\n" + count + " " + n_test + " " + (double)count/n_test);
	}
	
	void randomizeDataSet(Instances dataInstances, int numToBeKept) {
		Random r = new Random();
		
		dataInstances.randomize(r);
		System.out.println(dataInstances.numInstances());
		int temp2 =dataInstances.numInstances(); 
		for(int i=0; i <temp2-numToBeKept; i++) {
			int temp = r.nextInt(dataInstances.numInstances());
			dataInstances.delete(temp);
		}
		
		System.out.println(dataInstances.numInstances());
		
	}


	public static void main(String[] args) throws IOException {

		if(args.length != 3) {
			System.out.println("args != 3");
			return;
		}

		File datasetFile = new File(args[0]);
		ArffLoader arffLoader = new ArffLoader();
		arffLoader.setFile(datasetFile);

		Instances dataInstances = arffLoader.getDataSet();

		Ml3_NaiveBayes ml3 = new Ml3_NaiveBayes();
		ml3.randomizeDataSet(dataInstances, 50);
		ml3.setClassLabels(dataInstances);

		int m = dataInstances.numAttributes();

		for(int i = 0; i < dataInstances.numInstances(); i++) {
			if(dataInstances.instance(i).stringValue(m-1).equals(negClass)) {
				numOfNegClass ++;
			} else {
				numOfPosClass ++;
			}
		}


		File datasetFile2 = new File(args[1]);
		arffLoader = new ArffLoader();
		arffLoader.setFile(datasetFile2);

		Instances testInstances = arffLoader.getDataSet();

		if(args[2].equals("n")) {
			ml3.naiveBayes(dataInstances, testInstances);
		} else if(args[2].equals("t")) {
			ml3.tan(dataInstances, testInstances);
		}
	}
}
