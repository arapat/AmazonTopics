
import math.log
import util.Random

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{Vector, Vectors, SparseVector}

object AmazonCanopyClustering {
    val unit = 0.0001
    val lnOf2 = log(2)
    val stopwords = Array("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "ve", "ll")

    def computeJSDivergence(vec1: SparseVector, vec2: SparseVector) = {
      val indices = (vec1.indices ++ vec2.indices).distinct
      var result = 0.0
      for (i <- indices) {
        val a = vec1(i)
        val b = vec2(i)
        val m = (a + b) * 0.5
        if (a != 0) {
          result = result + a * log(a / m) / lnOf2
        }
        if (b != 0) {
          result = result + b * log(b / m) / lnOf2
        }
      }
      result / 2.0
    }

    def main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("AmazonCanopyClustering")
                                       .set("spark.executor.memory", "50g")
        val sc = new SparkContext(sparkConf)
        val lines = sc.textFile("/user/arapat/amazon/all.txt")
        // val lines = sc.textFile("/user/arapat/amazon/Electronics.txt")

        val sharedStopwords = sc.broadcast(stopwords)
        val reviews = lines.map(_.split(":")
                           .map(_.trim))
                           .filter(_(0) == "review/text")
                           .map(_(1))
        val reviewsCount = reviews.count
        val tokens = reviews.flatMap(line =>
                                "[a-zA-Z]+".r findAllIn line map (_.toLowerCase))
                            .filter(token =>
                                sharedStopwords.value.contains(token) == false)
        val wordsCount = tokens.map((_, unit.toDouble))
                               .reduceByKey(_ + _)
                               // .cache

        val pairsCount = reviews.map(line =>
              "[a-zA-Z]+".r findAllIn line map (_.toLowerCase))
            .flatMap(array =>
              for { a <- array; b <- array if (a != b) } yield (a, b))
            .filter {case (w1, w2) =>
              sharedStopwords.value.contains(w1) == false &&
              sharedStopwords.value.contains(w2) == false}
            .map((_, unit.toDouble))
            .reduceByKey(_ + _)
        val pairsSum = pairsCount.map(pair => (pair._1._1, pair._2))
                                 .reduceByKey(_ + _)

        // Compute the product of KL divergence and word counts 
        // for each words that appear more than 10 times
        val total = wordsCount.map(_._2).reduce(_ + _)
        val stat = pairsCount.map(a => (a._1._1, a))
                             .join(wordsCount)
                             .filter(_._2._2 >= 10 * unit)
                             .map(a => (a._1, a._2._1))
                             .join(pairsSum)
                             .map {case (w, (((ww, vv), pc), wps)) =>
                               (vv, (ww, wps, pc))}
                             .join(wordsCount)
                             .filter(_._2._2 >= 10 * unit)
                             .map {case (v, ((ww, wps, pc), vc)) =>
                               ((ww, v), (wps, vc, pc))}
                             // .cache
        val kld = stat.map {case ((ww, v), (wps, vc, pc)) =>
                              (ww, (wps, vc, pc, total))}
                      .map {case (w, (wps, vc, pc, tc)) =>
                              val div =
                                math.log(pc * tc / wps / vc) * pc / wps
                                (w, div)
                           }.reduceByKey(_ + _) //.sortBy(_._2, false)

        val dictList = stat.flatMap(_._1.productIterator.map(_.asInstanceOf[String]))
                           .distinct.collect
        val dictSize = dictList.size
        val dictMap = dictList.zipWithIndex.toMap
        val sharedDictMap = sc.broadcast(dictMap)
        val signature = kld.join(wordsCount)
                           .map {case (w, (div, count)) => (w, div * count)}
                           .filter(_._2 >= 40.0)
                           .map(_._1)
                           // .sortBy(_._2, false)
                           .collect
        val sharedSignature = sc.broadcast(signature)

        val preVector = stat.filter(item =>
                              sharedSignature.value.contains(item._1._1))
                            .map {case ((ww, v), (wps, vc, pc)) =>
                                    (ww, (Array(sharedDictMap.value(v)), Array(pc / wps)))}
        val sparseVectors = preVector.reduceByKey {
          (p1 : (Array[Int], Array[Double]), p2 : (Array[Int], Array[Double])) =>
            (p1._1 ++ p2._1, p1._2 ++ p2._2)
        }.map {case (ww, (index, value)) =>
                (ww, Vectors.sparse(dictSize, index, value))}.cache
        val pairWise = sparseVectors.cartesian(sparseVectors)
                                    .filter {case (vector1, vector2) =>
                                      vector1._1 < vector2._1}
        val jsd = pairWise.map {case (vector1, vector2) =>
          (vector1._1, vector2._1,
            computeJSDivergence(vector1._2.asInstanceOf[SparseVector],
              vector2._2.asInstanceOf[SparseVector]))}.cache

        var remainingPoints = sparseVectors.map(_._1).collect()
        println("Remain: " + remainingPoints.size.toString)
        val t1 = 0.02
        val t2 = 0.02
        while (remainingPoints.nonEmpty) {
          val sample = remainingPoints(Random.nextInt(remainingPoints.size))
          val dist = jsd.filter {case (w1, w2, c) =>
                                      (w1 == sample || w2 == sample)}
                        .collect
          val entropy = dist.filter(_._3 <= t1)
                            .flatMap(p => Array(p._1, p._2))
                            .filter(remainingPoints.contains(_))
                            .distinct
          println(sample + " -> [" + entropy.mkString(", ") + "]")
          val closePoints = dist.filter(_._3 <= t2)
                                .flatMap(p => Array(p._1, p._2))
                                .distinct ++ Array(sample)
          remainingPoints = remainingPoints.filterNot(p => closePoints.contains(p))
          println("Remain: " + remainingPoints.size.toString)
        }

        // println
        // println("Dict size: " + dictList.size)
        // println("Signature size: " + signature.size)

        // println(sparseVectors.count)
        // println(jsd.count)
        sc.stop()
        println
        println("Done.")
    }
}

