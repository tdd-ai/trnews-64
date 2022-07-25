# trnews-64 dataset

<a href="https://doi.org/10.5281/zenodo.5180654"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.5180654.svg" alt="DOI"></a>

__trnews-64__ is a character language modeling dataset that contain 64 million words of news articles and columns.
It can be utilized as a benchmark for different modeling long range dependencies in Turkish language.

This dataset contains a mix of news articles from different topics and journals from different domains retrieved from [TS Timeline Corpus](https://tscorpus.com/corpora/ts-timeline-corpus/).

## Preprocessing

This dataset was preprocessed and clean from infrequent characters. The main character set is shared in the file `tr.charset.json`, which contains 124 characters in total. This includes Turkish upper/lower case characters along with punctuations and some other common characters. 

## Download

The dataset is hosted on [Zenodo](https://zenodo.org/), it can be downloaded using the following:

```bash
wget -O trnews-64.tar.bz2 https://zenodo.org/record/5180654/files/trnews-64.tar.bz2?download=1
tar -xf trnews-64.tar.bz2
rm trnews-64.tar.bz2
```

## Details

Dataset splits are shared in raw text format and the articles are seperated by empty lines.

**Example**:

```
Dolar dün 2.5075 liraya kadar çıkarak rekor kırmasının ordından bugün 2.49 - 2.50 lira aralığında hareket etti. Cari işlemler açığının beklentilere paralel gelmesinin de etkisiyle 2.4820 liraya kadar çekilen dolar, daha sonra gelens alımlarla 2.5085'e çıkarak rekorunu tazeledi. ABD para birimi daha sonra 2.5050 - 2.5070 düzeylerinde hareket ederken, euro da 2.8380 lira düzeylerine çıktı ve yarı yarıya euro ve dolardan oluşan döviz sepeti de 2.63 düzeyinin üstüne çıktı.

DW Türkçe Servisi’nin aktardığına göre, ‘Aghet’ (Ağıt) konserinin Almanya’nın İstanbul Başkonsolosluğu’ndaki temsiline Cumhurbaşkanı Recep Tayyip Erdoğan da davet edildi. Alman haber ajansı dpa’nın haberinde, Erdoğan’ın yanı sıra Başbakan Binali Yıldırım, Dışişleri Bakanı Mevlüt Çavuşoğlu ile Kültür ve Turizm Bakanı Nabi Avcı’nın da davetliler arasında olduğu belirtildi. Habere göre, gönderilen davetiyelerde etkinlikte ‘Türk ve Ermeni geçmişlerindeki yaralar’ ile ifade ve sanat özgürlüğünün ele alınacağı ifade edildi. Dresden Senfoni Orkestrası tarafından hazırlanan ‘Aghet’ konseri, İstanbul Başkonsolosluğu’nda 13 Kasım’da gerçekleştirilecek. Etkinlikte ayrıca Türk-Ermeni-Alman Dostluk Derneği’nin kurulması planlanıyor.
```

This dataset is split based on the number of articles for each split. In the table below you can find some statistics:

|                                 | train  |  val  | test  | total  |
|---------------------------------|--------|-------|-------|--------|
| # of Articles                   | 140000 | 5000  | 5000  | 150000 |
| Raw text size                   | 402MBs | 15MBs | 15MBs | 431MBs |
| Total # of words                | 59.7M  | 2.1M  | 2.1M  |  64M   |
| Avg. # of words per article     |  426   |  424  | 420   |  426   |
| Avg. # of sentences per article |  23    |  22.8 | 22.9  |  23    |

## Splitting articles

To seperate articles with their titles you can use this snippet: 

```python
import re

with open("trnews-64.test.raw") as fi:
    articles = re.split("\n\n", fi.read()) 
```

## Citation

```
@inproceedings{safaya-etal-2022-mukayese,
    title = "Mukayese: {T}urkish {NLP} Strikes Back",
    author = "Safaya, Ali  and
      Kurtulu{\c{s}}, Emirhan  and
      Goktogan, Arda  and
      Yuret, Deniz",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.69",
    doi = "10.18653/v1/2022.findings-acl.69",
    pages = "846--863",
    abstract = "Having sufficient resources for language X lifts it from the under-resourced languages class, but not necessarily from the under-researched class. In this paper, we address the problem of the absence of organized benchmarks in the Turkish language. We demonstrate that languages such as Turkish are left behind the state-of-the-art in NLP applications. As a solution, we present Mukayese, a set of NLP benchmarks for the Turkish language that contains several NLP tasks. We work on one or more datasets for each benchmark and present two or more baselines. Moreover, we present four new benchmarking datasets in Turkish for language modeling, sentence segmentation, and spell checking. All datasets and baselines are available under: https://github.com/alisafaya/mukayese",
}
```

## License

This dataset is licensed under [Creative Commons Attribution 4.0 International](./LICENSE) license.

## Contact

Ali Safaya (alisafaya at gmail dot com).
