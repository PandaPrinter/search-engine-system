package edu.asu.irs13;

import org.apache.lucene.index.*;

import org.apache.lucene.store.*;
import org.apache.lucene.document.*;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;

public class MySearchEnginePart1 {

	static HashMap<Integer, Double> tfNorm = new HashMap<Integer, Double>();
	static HashMap<Integer, Double> tfIdfNorm = new HashMap<Integer, Double>();
	static HashMap<String, Double> termIdf = new HashMap<String, Double>();

	public static void tfNorm(IndexReader r) throws Exception {
		double begin = System.currentTimeMillis();
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
		double end = System.currentTimeMillis();
		double value = end - begin;
		System.out.println("Time spent on NormTF : " + value);
	}

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
			idf = Math.log(totalDoc / termDoc) / Math.log(2) * 1.0;
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

	public static void cosineTf(String str, IndexReader r) throws IOException {
		// double begin = System.currentTimeMillis();
		HashMap<Integer, Double> cosTftemp = new HashMap<Integer, Double>();
		HashMap<Integer, Double> cosTf = new HashMap<Integer, Double>();
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
		// long begin = System.nanoTime();
		for (String term : query.keySet()) {
			Term te = new Term("contents", term);
			double cos = (double) 0;
			TermDocs td = r.termDocs(te);
			while (td.next()) {
				if (cosTftemp.containsKey(td.doc())) {
					cos = cosTftemp.get(td.doc());
				}
				cos += td.freq() * query.get(term);
				cosTftemp.put(td.doc(), cos);
				cos = 0.0;
			}
		}
		// long end = System.nanoTime();
		// long value = end - begin;
		// System.out.println("Time spent on searching in cosineTf : " + value);

		for (String term : query.keySet()) {
			queryNorm += query.get(term) * query.get(term);
		}
		for (Integer doc : cosTftemp.keySet()) {
			double cos = cosTftemp.get(doc) / (tfNorm.get(doc) * Math.sqrt(queryNorm));
			cosTf.put(doc, cos);
		}
		// double end = System.currentTimeMillis();
		// double value = end - begin;
		// System.out.println("Time spent on cosineTf : " + value);
		sortResult(cosTf, r);
	}

	public static void cosineTfIdf(String str, IndexReader r) throws IOException {
		// long begin = System.nanoTime();
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
		// long begin = System.nanoTime();
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
		// long end = System.nanoTime();
		// long value = end - begin;
		// System.out.println("Time spent on searching in cosineTfIdf : " +
		// value);

		for (String term : query.keySet()) {
			queryNorm += query.get(term) * query.get(term);
		}
		for (Integer doc : cosTfIdftemp.keySet()) {
			double cos = cosTfIdftemp.get(doc) / (tfIdfNorm.get(doc) * Math.sqrt(queryNorm));
			cosTfIdf.put(doc, cos);
		}
		// double end = System.currentTimeMillis();
		// double value = end - begin;
		// System.out.println("Time spent on cosineTfIdf : " + value);
		sortResult(cosTfIdf, r);
		// long end = System.nanoTime();
		// long value = end - begin;
		// System.out.println("Time spent on searching in cosineTfIdf : " +
		// value);
	}

	public static void sortResult(HashMap<Integer, Double> map, IndexReader r) throws IOException {
		// double begin = System.currentTimeMillis();
		int count = 0;
		Set<Entry<Integer, Double>> set = map.entrySet();
		List<Entry<Integer, Double>> list = new ArrayList<Entry<Integer, Double>>(set);
		Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>() {
			public int compare(Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2) {
				return (o2.getValue()).compareTo(o1.getValue());
			}
		});
		System.out.println("Size of the results: " + map.size());
		for (Map.Entry<Integer, Double> entry : list) {
			count++;
			if (count <= 10) {
				// String d_url =
				// r.document(entry.getKey()).getFieldable("path").stringValue().replace("%%",
				// "/");
				// System.out.println(count + ". " + "[" + entry.getKey() + "] "
				// + " VALUE: " + entry.getValue());
				System.out.println(count + ". " + "[" + entry.getKey() + "]");
			} else
				break;
		}
		// double end = System.currentTimeMillis();
		// double value = end - begin;
		// System.out.println("Time spent on sortResult : " + value);
	}

	public static void sortIDF(HashMap<String, Double> map) throws IOException {
		int count = 0;
		Set<Entry<String, Double>> set = map.entrySet();
		List<Entry<String, Double>> list = new ArrayList<Entry<String, Double>>(set);
		Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
			public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
				return (o1.getValue()).compareTo(o2.getValue());
			}
		});
		for (Map.Entry<String, Double> entry : list) {
			count++;
			if (count <= 50) {
				System.out.println(count + ". " + "[" + entry.getKey() + "] " + "  VALUE: " + entry.getValue());
				// System.out.println(count + ". " + "[" + entry.getKey() + "]
				// ");
			} else
				break;
		}
	}

	public static void main(String[] args) throws Exception {
		IndexReader r = IndexReader.open(FSDirectory.open(new File("index")));
		tfNorm(r);
		tfIdfNorm(r);
		// sortIDF(termIdf);
		Scanner sc = new Scanner(System.in);
		String option;
		System.out.println("Please choose from 1, 2, 3");
		System.out.println("1. Cosine TF");
		System.out.println("2. Cosine TF/IDF");
		System.out.println("3. Quit");
		option = sc.nextLine();
		while (!option.equals("3") && !option.isEmpty()) {
			System.out.print("query> ");
			if (option.equals("1"))
				cosineTf(sc.nextLine(), r);
			else if (option.equals("2"))
				cosineTfIdf(sc.nextLine(), r);

			System.out.println("Please choose from 1, 2, 3");
			System.out.println("1. Cosine TF");
			System.out.println("2. Cosine TF/IDF");
			System.out.println("3. Quit");
			option = sc.nextLine();
		}
	}

}
