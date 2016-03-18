package edu.asu.irs13;

import org.apache.lucene.index.*;

import org.apache.lucene.store.*;
import org.apache.lucene.document.*;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;

public class Part2_PageRank {

	static HashMap<Integer, Double> tfNorm = new HashMap<Integer, Double>();
	static HashMap<Integer, Double> tfIdfNorm = new HashMap<Integer, Double>();
	static HashMap<String, Double> termIdf = new HashMap<String, Double>();
	private static final long MEGABYTE = 1024L * 1024L;
	// static HashMap<Integer, Double> cosTfIdf = new HashMap<Integer,
	// Double>();

	public static void tfNorm(IndexReader r) throws Exception {
		TermEnum t = r.terms();
		double frequency = (double) 0;
		while (t.next()) {
			Term te = new Term("contents", t.term().text());
			TermDocs td = r.termDocs(te);
			while (td.next()) {
				if (tfNorm.containsKey(td.doc())) {
					frequency = tfNorm.get(td.doc());
				}
				frequency += td.freq() * td.freq();
				tfNorm.put(td.doc(), frequency);
				frequency = (double) 0;
			}
		}
		for (Map.Entry<Integer, Double> entry : tfNorm.entrySet()) {
			double value = entry.getValue();
			value = Math.sqrt(value);
			tfNorm.put(entry.getKey(), value);
		}
	}

	public static void tfIdfNorm(IndexReader r) throws Exception {
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
	}

	public static void cosineTfIdf(Double w, String str, IndexReader r) throws Exception {
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
		// sortResult(cosTfIdf, r);
		getPageRank(w, r, cosTfIdf);
	}

	public static void getPageRank(Double w, IndexReader r, HashMap<Integer, Double> cosTfIdf) throws Exception {
		long start = System.nanoTime();

		double c = 0.8;
		int docNum = r.maxDoc();
		LinkAnalysis.numDocs = docNum;
		LinkAnalysis l = new LinkAnalysis();
		double[] Ri0 = new double[docNum];
		double[] Ri1 = new double[docNum];
		List<Integer> sinkPages = new ArrayList<Integer>();
		// find and store sink pages
		for (int i = 0; i < docNum; i++) {
			if (l.getLinks(i).length == 0) {
				sinkPages.add(i);
			}
		}
		// iteration counter
		int y = 0;
		double value = 1.0 / docNum;
		for (int i = 0; i < docNum; i++)
			Ri1[i] = value;
		double maxThreshold = 1.0;
		double threshold = 0.000000001;
		double[] linkMatrix = new double[docNum];
		while (maxThreshold > threshold) {
			y++;
			for (int i = 0; i < docNum; i++)
				Ri0[i] = Ri1[i];

			for (int j = 0; j < docNum; j++) {
				double temp = (1 - c) * value;
				Arrays.fill(linkMatrix, temp);

				int[] citations = l.getCitations(j);
				for (int ci : citations) {
					int sum = l.getLinks(ci).length;
					linkMatrix[ci] += (1.0 / sum) * c;

				}

				for (int m : sinkPages)
					linkMatrix[m] += c * value;

				double res = 0.0;
				for (int i = 0; i < docNum; i++) {
					res += linkMatrix[i] * Ri0[i];
				}
				Ri1[j] = res;
			}

			// normalization
			double sum = 0.0;
			for (double i : Ri1)
				sum += i;
			for (int i = 0; i < docNum; i++)
				Ri1[i] = Ri1[i] / sum;

			double atemp = Math.abs(Ri1[0] - Ri0[0]);
			for (int i = 1; i < docNum; i++) {
				double current = Math.abs(Ri1[i] - Ri0[i]);
				if (atemp < current)
					atemp = current;
			}
			maxThreshold = atemp;
			// System.out.println("Current max threshold value: " +
			// maxThreshold);
		}

		// normalize R[d] to make it compatible with vector space similarity
		double[] tempArr = new double[docNum];
		System.arraycopy(Ri0, 0, tempArr, 0, Ri0.length);
		Arrays.sort(tempArr);

		for (int i = 0; i < docNum; i++) {
			Ri0[i] = (Ri0[i] - tempArr[0]) / (tempArr[docNum - 1] - tempArr[0]);
		}
		// combine with cosine similarity
		HashMap<Integer, Double> finalRes = new HashMap<Integer, Double>();
		for (int i = 0; i < docNum; i++) {
			if (cosTfIdf.get(i) != null) {
				double finalres = w * Ri0[i] + (1 - w) * cosTfIdf.get(i);
				finalRes.put(i, finalres);
			}
		}

		// Test Authorities, Hubs results and Tf/Idf results
		// int[] authoritiesRes = {924, 24024, 23671, 24082, 24191, 2283, 24052,
		// 24105, 24113, 24166};
		// int[] hubsRes = {24092, 24064, 24061, 24082, 24108, 24093, 24193,
		// 24054, 24189, 24117};
		// int[] tfidf = {22156, 233, 19822, 19590, 22149, 22913, 21047, 1851,
		// 22936, 18699};
		// System.out.println("Authorities results: ");
		// for (int i : authoritiesRes){
		// if (finalRes.containsKey(i)){
		// System.out.println(i + ": " + finalRes.get(i));
		// }
		// }
		// System.out.println();
		// System.out.println("Hubs results: ");
		// for (int i : hubsRes){
		// if (finalRes.containsKey(i)){
		// System.out.println(i + ": " + finalRes.get(i));
		// }
		// }
		// System.out.println();
		// System.out.println("Tf/Idf results: ");
		// for (int i : tfidf){
		// if (finalRes.containsKey(i)){
		// System.out.println(i + ": " + finalRes.get(i));
		// }
		// }

		// sort results
		int count2 = 0;
		Map<Integer, Double> sortedMap = sortByValue(finalRes);

		System.out.println("Top 10 PageRank output:");
		for (Map.Entry<Integer, Double> entry : sortedMap.entrySet()) {
			if (count2 < 10) {
				Document d = r.document(entry.getKey());
				String url = d.getFieldable("path").stringValue();
				System.out.println(entry.getKey() + ": " + entry.getValue() + ": " + url.replace("%%", "/"));
				count2++;
			}
		}
		System.out.println("time of iteration is: " + y);
		long end = System.nanoTime();
		long value1 = end - start;
		System.out.println("Total time spent : " + value1);
		// Get the Java runtime
		Runtime runtime = Runtime.getRuntime();
		// Run the garbage collector
		runtime.gc();
		// Calculate the used memory
		long memory = runtime.totalMemory() - runtime.freeMemory();
		System.out.println("Used memory is bytes: " + memory);
		System.out.println("Used memory is megabytes: " + bytesToMegabytes(memory));
	}

	public static Map sortByValue(Map unsortedMap) {
		Map sortedMap = new TreeMap(new VComparator(unsortedMap));
		sortedMap.putAll(unsortedMap);
		return sortedMap;
	}

	public static long bytesToMegabytes(long bytes) {
		return bytes / MEGABYTE;
	}

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		IndexReader r = IndexReader.open(FSDirectory.open(new File("index")));
		tfNorm(r);
		tfIdfNorm(r);
		Scanner sc = new Scanner(System.in);
		String option;
		System.out.println("Please choose from 1, 2");
		System.out.println("1. PageRank");
		System.out.println("2. Quit");
		option = sc.nextLine();
		while (!option.equals("2") && !option.isEmpty()) {
			if (option.equals("1")) {
				System.out.print("query> ");
				String query = sc.nextLine();
				System.out.print("Please Enter w value: ");
				double w = Double.valueOf(sc.nextLine());
				if (w > 0 && w < 1) {
					cosineTfIdf(w, query, r);
					// getPageRank(w, r);
				} else
					System.out.println("Invalid w value!");
			}
			System.out.println("Please choose from 1, 2");
			System.out.println("1. PageRank");
			System.out.println("2. Quit");
			option = sc.nextLine();
		}
	}
}

class VComparator implements Comparator {
	Map map;

	public VComparator(Map map) {
		this.map = map;
	}

	public int compare(Object keyA, Object keyB) {
		Comparable valueA = (Comparable) map.get(keyA);
		Comparable valueB = (Comparable) map.get(keyB);
		return valueB.compareTo(valueA);
	}
}
