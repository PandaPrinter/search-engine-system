// Random Partition method

package edu.asu.irs13;

import java.io.File;

import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermDocs;
import org.apache.lucene.index.TermEnum;
import org.apache.lucene.store.FSDirectory;

public class KMeans {

	static HashMap<Integer, Double> tfIdfNorm = new HashMap<Integer, Double>();
	static HashMap<String, Double> termIdf = new HashMap<String, Double>();

	public static void tfIdfNorm(IndexReader r) throws Exception {
		double begin = System.currentTimeMillis();
		TermEnum t = r.terms();
		double tfIdf = (double) 0;
		double idf = 0.0;
		while (t.next()) {
			Term te = new Term("contents", t.term().text());
			TermDocs td = r.termDocs(te);
			int termDoc = r.docFreq(t.term());
			int totalDoc = r.maxDoc();
			idf = Math.log(totalDoc / termDoc) * 1.0;
			termIdf.put(t.term().text(), idf);
			while (td.next()) {
				if (tfIdfNorm.containsKey(td.doc())) {
					tfIdf = tfIdfNorm.get(td.doc());
				}
				tfIdf += td.freq() * idf * td.freq() * idf;
				tfIdfNorm.put(td.doc(), tfIdf);
				tfIdf = 0.0;
			}
			idf = 0.0;
		}
		for (Map.Entry<Integer, Double> entry : tfIdfNorm.entrySet()) {
			double value = entry.getValue();
			value = Math.sqrt(value);
			tfIdfNorm.put(entry.getKey(), value);
		}
		double end = System.currentTimeMillis();
		double value = end - begin;
		System.out.println("Time spent on NormTFIDF : " + value);
	}

	public static Integer[] cosineTfIdf(String str, IndexReader r) throws IOException {
		HashMap<Integer, Double> cosTfIdftemp = new HashMap<Integer, Double>();
		HashMap<Integer, Double> cosTfIdf = new HashMap<Integer, Double>();
		HashMap<String, Integer> query = new HashMap<String, Integer>();
		String[] terms = str.split("\\s+");
		int queryNorm = 0;
		for (String term : terms) {
			int frequency = 0;
			if (query.containsKey(term)) {
				frequency = query.get(term);
			}
			frequency++;
			query.put(term, frequency);
		}
		for (String term : query.keySet()) {
			Term te = new Term("contents", term);
			double cos = (double) 0;
			TermDocs td = r.termDocs(te);
			while (td.next()) {
				if (cosTfIdftemp.containsKey(td.doc())) {
					cos = cosTfIdftemp.get(td.doc());
				}
				cos += td.freq() * query.get(term) * termIdf.get(term);
				cosTfIdftemp.put(td.doc(), cos);
				cos = 0.0;
			}
		}
		for (String term : query.keySet()) {
			queryNorm += query.get(term) * query.get(term);
		}
		for (Integer doc : cosTfIdftemp.keySet()) {
			double cos = cosTfIdftemp.get(doc) / (tfIdfNorm.get(doc) * Math.sqrt(queryNorm));
			cosTfIdf.put(doc, cos);
		}
		return sortResult(cosTfIdf, r);
	}

	public static Integer[] sortResult(HashMap<Integer, Double> map, IndexReader r) throws IOException {
		int count = 0;
		Integer[] docs = new Integer[50];
		Set<Entry<Integer, Double>> set = map.entrySet();
		List<Entry<Integer, Double>> list = new ArrayList<Entry<Integer, Double>>(set);
		Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>() {
			public int compare(Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2) {
				return (o2.getValue()).compareTo(o1.getValue());
			}
		});
		for (Map.Entry<Integer, Double> entry : list) {
			if (count < 50) {
				docs[count] = entry.getKey();
				count++;
			} else {
				break;
			}
		}
		return docs;
	}

	public static void getKMeans(int k, String str, IndexReader r) throws IOException {

		List<Integer> base = Arrays.asList(cosineTfIdf(str, r));
		int baseCount = base.size();
		int[] ranNum = new int[k - 1];
		HashMap<Integer, Double[]> docsInfo = new HashMap<Integer, Double[]>();
		HashMap<Integer, List<Integer>> clusterInfo = new HashMap<Integer, List<Integer>>();

		TermEnum t = r.terms();

		// generate word base for these 50 docs
		HashSet<String> wordSet = new HashSet<String>();
		while (t.next()) {
			Term te = new Term("contents", t.term().text());
			TermDocs td = r.termDocs(te);
			while (td.next()) {
				if (base.contains(td.doc())) {
					wordSet.add(t.term().text());
					break;
				}
			}
		}

		int size = wordSet.size();

		// initialize docsInfo
		for (int i = 0; i < baseCount; i++) {
			Double[] tempArr = new Double[size];
			Arrays.fill(tempArr, 0.0);
			docsInfo.put(base.get(i), tempArr);
		}

		// update docsInfo with tf/idf value
		int index = 0;
		List<String> wordList = new ArrayList<String>(wordSet);
		for (String word : wordList) {
			Term te = new Term("contents", word);
			TermDocs td = r.termDocs(te);
			while (td.next()) {
				if (base.contains(td.doc())) {
					double tfIdf = td.freq() * termIdf.get(word);
					docsInfo.get(td.doc())[index] = tfIdf;
				}
			}
			index++;
		}
		
		// normalization
		for (Map.Entry<Integer, Double[]> e : docsInfo.entrySet()) {
			Double[] temp = e.getValue();
			double maxValue = temp[0];
			for (int i = 0; i < temp.length; i++) {
				if (temp[i] > maxValue) {
					maxValue = temp[i];
				}
			}
			for (int i = 0; i < temp.length; i++) {
				double tempValue = docsInfo.get(e.getKey())[i];
				tempValue = tempValue / maxValue;
				docsInfo.get(e.getKey())[i] = tempValue;
			}
		}
		
		// generate k-1 random numbers and store them into the list
		ArrayList<Integer> list = new ArrayList<Integer>();
		for (int i = 1; i < 50; i++) {
			list.add(new Integer(i));
		}
		Collections.shuffle(list);
		for (int i = 0; i < k - 1; i++) {
			ranNum[i] = list.get(i);
		}
		Arrays.sort(ranNum);

		// initialize centroidInfo
		ArrayList<Double[]> centroidInfo = new ArrayList<Double[]>(k);
		for (int i = 0; i < k; i++) {
			Double[] tempArr2 = new Double[size];
			Arrays.fill(tempArr2, 0.0);
			centroidInfo.add(tempArr2);
		}

		// generate k clusters and store them into clusterInfo
		List<Integer> tempCluster;
		int count = 0;
		for (int i = 0; i < k; i++) {
			if (i != k - 1) {
				tempCluster = new ArrayList<Integer>(base.subList(count, ranNum[i]));
				// store index of each doc in base into clusterInfo
				clusterInfo.put(i, tempCluster);
				count = ranNum[i];
			} else {
				tempCluster = new ArrayList<Integer>(base.subList(count, 50));
				clusterInfo.put(i, tempCluster);
			}
		}

		// calculate centroid
		for (int i = 0; i < k; i++) {
			double sum = 0.0;
			for (int j = 0; j < size; j++) {
				for (int docId : clusterInfo.get(i)) {
					sum += docsInfo.get(docId)[j];
				}
				sum = sum / clusterInfo.get(i).size();
				centroidInfo.get(i)[j] = sum;
				sum = 0.0;
			}
		}

		int iterationCount = 0;
		boolean flag = true;
		HashMap<Integer, Double> similarityRes = new HashMap<Integer, Double>();
		HashMap<Double, Integer> tempSimilarityRes = new HashMap<Double, Integer>();

		// begin iteration
		while (flag) {
			similarityRes.clear();
			tempSimilarityRes.clear();
			
			ArrayList<Double[]> newCentroidInfo = new ArrayList<Double[]>(centroidInfo);
			HashMap<Integer, List<Integer>> newClusterInfo = new HashMap<Integer, List<Integer>>(clusterInfo);
			
			Double[] compareRes = new Double[k];
			Arrays.fill(compareRes, 0.0);

			for (int i : base) {

				int clusterId = 0;
				Double[] value = docsInfo.get(i);

				// find the original cluster Id of this doc
				for (int j = 0; j < k; j++) {
					if (clusterInfo.get(j).contains(i)) {
						clusterId = j;
					}
				}

				// calculate the distance between the current doc and each
				// centroid
				for (int j = 0; j < k; j++) {
					Double[] centroid = newCentroidInfo.get(j);
					compareRes[j] = cosineCompareDistance(centroid, value);
				}

				// get the index of minimum value in the compareRes
				List<Double> temp = new ArrayList<Double>(Arrays.asList(compareRes));
				Arrays.sort(compareRes);
				int newClusterId = temp.indexOf(compareRes[k - 1]);
				similarityRes.put(i, compareRes[k - 1]);
				tempSimilarityRes.put(compareRes[k - 1], i);
				
				// when the doc changes its clusterId
				if (clusterId != newClusterId) {
					flag = false;
					newClusterInfo.get(newClusterId).add(new Integer(i));
					newClusterInfo.get(clusterId).remove(new Integer(i));
				}
			}

			// recalculate the centroids for each cluster
			if (flag == false) {
				double tempSum = 0.0;
				for (int i = 0; i < k; i++) {
					List<Integer> tempList = newClusterInfo.get(i);
					for (int j = 0; j < size; j++) {
						for (int e : tempList) {
							tempSum += docsInfo.get(e)[j];
						}
						tempSum = tempSum / tempList.size();
						newCentroidInfo.get(i)[j] = tempSum;
						tempSum = 0.0;
					}
				}
				flag = true;
				clusterInfo = new HashMap<Integer, List<Integer>>(newClusterInfo);
				centroidInfo = new ArrayList<Double[]>(newCentroidInfo);
			}
			// quit the iteration
			else {
				flag = false;
			}
			iterationCount++;
		}

		iterationCount--;
		System.out.println("iteration times: " + iterationCount);

		// show top-3 documents in each cluster
		for (Map.Entry<Integer, List<Integer>> cluster : clusterInfo.entrySet()) {
			ArrayList<Double> tempList = new ArrayList<Double>();
			int key = cluster.getKey();
			List<Integer> value = cluster.getValue();
			if (value.size() > 3) {
				for (int v : value){
					tempList.add(similarityRes.get(v));
				}
				Collections.sort(tempList);
				ArrayList<Integer> finalRes = new ArrayList<Integer>();
				for (int i = 0; i < 3; i++){
					finalRes.add(tempSimilarityRes.get(tempList.get(i)));
				}
				System.out.println("Cluster " + key + " : " + finalRes);
			} else {
				System.out.println("Cluster " + key + " : " + value);
			}
		}
	}

	/*********
	 * public static Double compareDistance(Double[] centroid, Double[] value) {
	 * double sum = 0.0; for (int i = 0; i < centroid.length; i++) { sum +=
	 * Math.pow((centroid[i] - value[i]), 2); } sum = Math.sqrt(sum); return
	 * sum; }
	 **********/

	public static Double cosineCompareDistance(Double[] centroid, Double[] value) {
		double sum = 0.0;
		double centroidSum = 0.0;
		double valueSum = 0.0;
		for (int i = 0; i < centroid.length; i++) {
			sum += centroid[i] * value[i];
			centroidSum += centroid[i] * centroid[i];
			valueSum += value[i] * value[i];
		}
		double temp = Math.sqrt(centroidSum) * Math.sqrt(valueSum);
		sum = sum / temp;
		return sum;
	}

	public static void main(String[] args) throws Exception {
		IndexReader r = IndexReader.open(FSDirectory.open(new File("index")));
		tfIdfNorm(r);
		Scanner sc = new Scanner(System.in);
		String option;
		System.out.println("Please choose from 1, 2");
		System.out.println("1. KMeans");
		System.out.println("2. Quit");
		option = sc.nextLine();
		while (!option.equals("2") && !option.isEmpty()) {
			if (option.equals("1")) {
				System.out.print("query> ");
				String query = sc.nextLine();
				System.out.print("Please Enter cluster number: ");
				int k = Integer.valueOf(sc.nextLine());
				if (k > 0) {
					getKMeans(k, query, r);
				} else
					System.out.println("Invalid value!");
			}
			System.out.println("Please choose from 1, 2");
			System.out.println("1. KMeans");
			System.out.println("2. Quit");
			option = sc.nextLine();
		}
	}
}
