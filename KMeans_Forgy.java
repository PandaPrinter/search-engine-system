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

public class KMeans_Forgy {

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
		int[] ranNum = new int[k];
		HashMap<Integer, HashMap<String, Double>> docsInfo = new HashMap<Integer, HashMap<String, Double>>();
		HashMap<Integer, Integer> maxFreq = new HashMap<Integer, Integer>();
		TermEnum t = r.terms();

		// initialize docsInfo
		for (int i = 0; i < baseCount; i++) {
			docsInfo.put(base.get(i), new HashMap<String, Double>());
			maxFreq.put(base.get(i), 0);
		}

		// update docsInfo with tf/idf value
		while (t.next()) {
			Term te = new Term("contents", t.term().text());
			TermDocs td = r.termDocs(te);
			while (td.next()) {
				if (docsInfo.containsKey(td.doc())) {
					String word = t.term().text();
					docsInfo.get(td.doc()).put(word, td.freq() * termIdf.get(word));
					if (td.freq() > maxFreq.get(td.doc())) {
						maxFreq.put(td.doc(), td.freq());
					}
				}
			}
		}

		// normalization
		for (int docId : base) {
			for (Map.Entry<String, Double> entry : docsInfo.get(docId).entrySet()) {
				double tempValue = docsInfo.get(docId).get(entry.getKey());
				tempValue = tempValue / maxFreq.get(docId);
				docsInfo.get(docId).put(entry.getKey(), tempValue);
			}
		}

		// generate k random numbers and store them into the list
		ArrayList<Integer> list = new ArrayList<Integer>();
		for (int i = 0; i < 50; i++) {
			list.add(new Integer(i));
		}
		Collections.shuffle(list);
		for (int i = 0; i < k; i++) {
			ranNum[i] = list.get(i);
		}

		// store centroid
		HashMap<Integer, HashMap<String, Double>> centroidInfo = new HashMap<Integer, HashMap<String, Double>>();
		for (int i = 0; i < k; i++) {
			centroidInfo.put(i, docsInfo.get(base.get(ranNum[i])));
		}

		int iterationCount = 0;
		boolean flag = true;

		int[] clusterInfo = new int[baseCount];
		Arrays.fill(clusterInfo, -1);

		// begin iteration
		while (flag) {

			double[][] compareRes = new double[k][baseCount];
			flag = false;

			// calculate centroid norm
			double[] centroidNorm = new double[k];
			for (int i = 0; i < k; i++) {
				double tempSum = 0.0;
				for (Map.Entry<String, Double> entry : centroidInfo.get(i).entrySet()) {
					double tempValue = entry.getValue();
					tempSum += tempValue * tempValue;
				}
				centroidNorm[i] = Math.sqrt(tempSum);
			}

			// calculate vector similarity between each document and each
			// centroid
			for (int i = 0; i < k; i++) {
				for (Map.Entry<String, Double> entry : centroidInfo.get(i).entrySet()) {
					for (int j = 0; j < baseCount; j++) {
						if (docsInfo.get(base.get(j)).containsKey(entry.getKey()))
							compareRes[i][j] += entry.getValue() * docsInfo.get(base.get(j)).get(entry.getKey());
					}
				}

				for (int j = 0; j < baseCount; j++) {
					compareRes[i][j] /= (centroidNorm[i] * tfIdfNorm.get(base.get(j)));
				}
			}

			int[] clusterDocCount = new int[k];

			// assign each document to responding centroid with maximum
			// similarity value
			for (int j = 0; j < baseCount; j++) {
				double maxVaule = 0.0;
				int index = 0;
				for (int i = 0; i < k; i++) {
					if (maxVaule < compareRes[i][j]) {
						maxVaule = compareRes[i][j];
						index = i;
					}
				}
				// compare previous cluster id with new cluster id
				if (clusterInfo[j] != index) {
					flag = true;
				}
				clusterInfo[j] = index;
				clusterDocCount[index]++;
			}

			if (flag == false) {
				break;
			}

			for (int i = 0; i < k; i++) {
				centroidInfo.put(i, new HashMap<String, Double>());
			}

			// recalculate centroid
			double value = 0.0;
			for (int i = 0; i < baseCount; i++) {
				for (Map.Entry<String, Double> entry : docsInfo.get(base.get(i)).entrySet()) {
					if (centroidInfo.get(clusterInfo[i]).containsKey(entry.getKey())) {
						double tempValue = centroidInfo.get(clusterInfo[i]).get(entry.getKey());
						tempValue += entry.getValue() / clusterDocCount[clusterInfo[i]];
						centroidInfo.get(clusterInfo[i]).put(entry.getKey(), tempValue);
					}

					else {
						value = entry.getValue() / clusterDocCount[clusterInfo[i]];
						centroidInfo.get(clusterInfo[i]).put(entry.getKey(), value);
					}
				}
			}

			iterationCount++;
		}

		System.out.println("iteration times: " + iterationCount);

		// show top-3 documents in each cluster
		for (int i = 0; i < k; i++) {
			StringBuilder sb = new StringBuilder();
			int count = 0;
			sb.append("Cluster " + i + ": ");
			for (int j = 0; j < baseCount; j++) {
				if (clusterInfo[j] == i) {
					// System.out.println(base.get(j));
					sb.append(base.get(j) + ", ");
					// Document d = r.document(bs[j]);
					// String url = d.getFieldable("path").stringValue(); // the
					// 'path' field of the Document object holds the URL
					// String url1 = url.replace("%%", "/");
					// System.out.println(url1);
					count++;
				}
				if (count == 3) {
					break;
				}
			}
			System.out.println(sb.toString());
		}
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
