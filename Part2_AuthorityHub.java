package edu.asu.irs13;

import org.apache.lucene.index.*;

import org.apache.lucene.store.*;
import org.apache.lucene.document.*;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;

public class Part2_AuthorityHub {

	static HashMap<Integer, Double> tfNorm = new HashMap<Integer, Double>();
	static HashMap<Integer, Double> tfIdfNorm = new HashMap<Integer, Double>();
	static HashMap<String, Double> termIdf = new HashMap<String, Double>();
	//static HashSet<Integer> baseSet = new HashSet<Integer>();

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

	public static void cosineTfIdf(String str, IndexReader r) throws IOException {
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
		sortResult(cosTfIdf, r);
	}

	public static void sortResult(HashMap<Integer, Double> map, IndexReader r) throws IOException {
		// double begin = System.currentTimeMillis();
		Set<Entry<Integer, Double>> set = map.entrySet();
		List<Entry<Integer, Double>> list = new ArrayList<Entry<Integer, Double>>(set);
		Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>() {
			public int compare(Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2) {
				return (o2.getValue()).compareTo(o1.getValue());
			}
		});
		getAuthoritiesHubs(list, r);
	}

	public static void getAuthoritiesHubs(List<Entry<Integer, Double>> list, IndexReader r) throws IOException {
		long begin1 = System.nanoTime();
		
		//generate base set
		HashSet<Integer> baseSet = new HashSet<Integer>();
		LinkAnalysis.numDocs = r.maxDoc();
		LinkAnalysis l = new LinkAnalysis();
		int count = 0;
		for (Map.Entry<Integer, Double> entry : list) {
			count++;
			int id = entry.getKey();
			if (count <= 10) {
				baseSet.add(id);
				int[] citations = l.getCitations(id);
				int[] links = l.getLinks(id);
				for (int c : citations) {
					baseSet.add(c);
				}
				for (int link : links)
					baseSet.add(link);
			} else
				break;
		}
		
		long end1 = System.nanoTime();
		long value1 = end1 - begin1;
		System.out.println("Time spent on generating base set : " + value1);
		
		// generate adjacency matrix and transpose matrix
		List<Integer> baseList = new ArrayList<Integer>(baseSet);
		double[][] matrix = new double[baseSet.size()][baseSet.size()];
		for (int i = 0; i < baseList.size(); i++) {
			int id = baseList.get(i);
			int[] linkSet = l.getLinks(id);
			for (int ll : linkSet) {
				if (baseList.contains(ll)) {
					matrix[i][baseList.indexOf(ll)] = 1.0;
				}
			}
		}

		long end5 = System.nanoTime();
		long value5 = end5 - end1;
		System.out.println("Time spent on generating adjacency matrix : " + value5);
		
		double[][] transposeMatrix = new double[baseSet.size()][baseSet.size()];
		for (int i = 0; i < baseSet.size(); i++) {
			for (int j = 0; j < baseSet.size(); j++) {
				transposeMatrix[j][i] = matrix[i][j];
			}
		}
		
		long end2 = System.nanoTime();
		long value2 = end2 - end1;
		System.out.println("Time spent on generating transpose matrix : " + value2);
		
		// iteration of computing authorities and hubs
		double[] authorities1 = new double[baseSet.size()]; // ai
		double[] authorities0 = new double[baseSet.size()]; // ai-1
		double[] hubs1 = new double[baseSet.size()]; // hi
		double[] hubs0 = new double[baseSet.size()]; // hi-1

		double maxThreshold = 1.0;
		double threshold = 0.000000001;

		double authitemp = 0.0;
		double hubsitemp = 0.0;

		for (int i = 0; i < baseSet.size(); i++) {
			authorities1[i] = 1.0;
			hubs1[i] = 1.0;
		}

		while (maxThreshold > threshold) {
			for (int i = 0; i < baseSet.size(); i++) {
				authorities0[i] = authorities1[i];
				hubs0[i] = hubs1[i];
			}

			for (int i = 0; i < baseSet.size(); i++) {
				for (int j = 0; j < baseSet.size(); j++) {
					double temp = hubs0[j];
					double mvalue = transposeMatrix[i][j];
					authitemp += temp * mvalue;
				}
				authorities1[i] = authitemp;
				authitemp = 0.0;
			}

			for (int i = 0; i < baseSet.size(); i++) {
				for (int j = 0; j < baseSet.size(); j++) {
					double temp = authorities1[j];
					double mvalue = matrix[i][j];
					hubsitemp += temp * mvalue;
				}
				hubs1[i] = hubsitemp;
				hubsitemp = 0.0;
			}

			// normalization part

			double totalValue = 0.0;
			for (int i = 0; i < baseSet.size(); i++)
				totalValue += authorities1[i] * authorities1[i];
			for (int i = 0; i < authorities1.length; i++)
				authorities1[i] = authorities1[i] / Math.sqrt(totalValue);
			totalValue = 0.0;
			for (int i = 0; i < baseSet.size(); i++)
				totalValue += hubs1[i] * hubs1[i];
			for (int i = 0; i < hubs1.length; i++)
				hubs1[i] = hubs1[i] / Math.sqrt(totalValue);

			double atemp = Math.abs(authorities1[0] - authorities0[0]);
			double htemp = Math.abs(hubs1[0] - hubs0[0]);

			for (int i = 1; i < baseSet.size(); i++) {
				double current = Math.abs(authorities1[i] - authorities0[i]);
				if (atemp < current)
					atemp = current;
				current = Math.abs(hubs1[i] - hubs0[i]);
				if (htemp < current)
					htemp = current;
			}

			maxThreshold = Math.max(atemp, htemp);

		}
		
		long end3 = System.nanoTime();
		long value3 = end3 - end2;
		System.out.println("Time spent on calculating AuthoritiesHubs : " + value3);
		
		// generate top 10 results

		HashMap<Integer, Double> tempAMap = new HashMap<Integer, Double>();
		HashMap<Integer, Double> tempHMap = new HashMap<Integer, Double>();
		for (int i = 0; i < baseSet.size(); i++) {
			tempAMap.put(i, authorities1[i]);
			tempHMap.put(i, hubs1[i]);
		}
		int count2 = 0;
		Map<Integer, Double> sortedAMap = sortByValue(tempAMap);
		Map<Integer, Double> sortedHMap = sortByValue(tempHMap);
		System.out.println("Top 10 authorities output:");
		for (Map.Entry<Integer, Double> entry : sortedAMap.entrySet()) {
			if (count2 < 10) {
				System.out.println(entry.getKey() + ": " + entry.getValue());
				count2++;
			}
		}
		count2 = 0;
		System.out.println();
		System.out.println("Top 10 hubs output:");
		for (Map.Entry<Integer, Double> entry : sortedHMap.entrySet()) {
			if (count2 < 10) {
				System.out.println(entry.getKey() + ": " + entry.getValue());
				count2++;
			}
		}
		System.out.println();
		
		long end4 = System.nanoTime();
		long value4 = end4 - end3;
		System.out.println("Time spent on generating top 10 results : " + value4);
	}

	public static Map sortByValue(Map unsortedMap) {
		Map sortedMap = new TreeMap(new ValueComparator(unsortedMap));
		sortedMap.putAll(unsortedMap);
		return sortedMap;
	}

	public static void main(String[] args) throws Exception {
		IndexReader r = IndexReader.open(FSDirectory.open(new File("index")));
		tfNorm(r);
		tfIdfNorm(r);
		Scanner sc = new Scanner(System.in);
		String option;
		System.out.println("Please choose from 1, 2");
		System.out.println("1. Get Authorities/Hubs");
		System.out.println("2. Quit");
		option = sc.nextLine();
		while (!option.equals("2") && !option.isEmpty()) {
			System.out.print("query> ");
			if (option.equals("1"))
				cosineTfIdf(sc.nextLine(), r);
			System.out.println("Please choose from 1, 2");
			System.out.println("1. Get Authorities/Hubs");
			System.out.println("2. Quit");
			option = sc.nextLine();
		}
	}
}

class ValueComparator implements Comparator {
	Map map;

	public ValueComparator(Map map) {
		this.map = map;
	}

	public int compare(Object keyA, Object keyB) {
		Comparable valueA = (Comparable) map.get(keyA);
		Comparable valueB = (Comparable) map.get(keyB);
		return valueB.compareTo(valueA);
	}
}
