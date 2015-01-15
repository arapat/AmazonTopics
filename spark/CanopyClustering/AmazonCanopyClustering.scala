
import math.log
import math.max
import math.min
import util.Random

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{Vector, Vectors, SparseVector}

object AmazonCanopyClustering {
    val stopwords = Array("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "ve", "ll", "haven", "hasn", "don", "doesn")

    def computeJSDivergence(vec1: SparseVector, vec2: SparseVector) = {

      val p1 = (vec1.indices zip vec1.values).sortBy(_._1)
      val len1 = p1.size

      val p2 = (vec2.indices zip vec2.values).sortBy(_._1)
      val len2 = p2.size

      var itr1 = 0
      var itr2 = 0
      var result = 0.0

      while (itr1 < len1 && itr2 < len2) {
        if (p1(itr1)._1 < p2(itr2)._1) {
          result = result + p1(itr1)._2 * log(2)
          itr1 = itr1 + 1
        } else if (p1(itr1)._1 > p2(itr2)._1) {
          result = result + p2(itr2)._2 * log(2)
          itr2 = itr2 + 1
        } else {
          val m = (p1(itr1)._2 + p2(itr2)._2) / 2.0
          result = result + p1(itr1)._2 * log(p1(itr1)._2 / m) + p2(itr2)._2 * log(p2(itr2)._2 / m)
          itr1 = itr1 + 1
          itr2 = itr2 + 1
        }
      }
      while (itr1 < len1) {
          result = result + p1(itr1)._2 * log(2)
          itr1 = itr1 + 1
      }
      while (itr2 < len2) {
          result = result + p2(itr2)._2 * log(2)
          itr2 = itr2 + 1
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

        // val ex = reviews.flatMap(line => line.split('.'))
        //                 .map(_.toLowerCase)
        //                 .filter(s => s.containsSlice("first") && s.containsSlice("well") )
        //                 .collect
        // ex.foreach(println)
        // return

        val tokens = reviews.flatMap(line => line.split('.'))
                            .flatMap(line =>
                                "[a-zA-Z']+".r findAllIn line map (_.toLowerCase))
                            .filter(token =>
                                sharedStopwords.value.contains(token) == false)
        val wordsCount = tokens.map((_, 1))
                               .reduceByKey(_ + _)
                               .cache

        // word pair will appear twice: (w, v) and (v, w)
        // word w and v should each appear at least 10 times
        val pairsCount = reviews.map(line =>
              "[a-zA-Z]+".r findAllIn line map (_.toLowerCase))
            .flatMap(array =>
              for { a <- array; b <- array if (a != b) } yield (a, b))
            .filter {case (w1, w2) =>
              sharedStopwords.value.contains(w1) == false &&
              sharedStopwords.value.contains(w2) == false}
            .map((_, 1.0)).reduceByKey(_ + _)
            .map(p => (p._1._1, p)).join(wordsCount).filter(_._2._2 >= 10)
            .map(_._2._1)
            .map(p => (p._1._2, p)).join(wordsCount).filter(_._2._2 >= 10)
            .map(_._2._1)
            .cache
        val pairsSum = pairsCount.map(p => (p._1._1, p._2))
                                 .reduceByKey(_ + _)

        // Compute the product of KL divergence and word counts 
        // for each words that appear more than 10 times
        val stat = pairsCount.map(p => (p._1._1, p))
                             .join(pairsSum)
                             .map { case (w, (((ww, vv), pc), sum_pc)) =>
                              (vv, ((ww, vv), pc, sum_pc))}
                             .join(wordsCount)
                             .map { case (v, (((ww, vv), pc, sum_pc), vc)) =>
                              ((ww, vv), (sum_pc, vc, pc))}
                             .cache
        val sumWordsCount = wordsCount.map(_._2).reduce(_ + _)
        val kld = stat.map {case ((ww, vv), (sum_pc, vc, pc)) =>
          val div = math.log(pc * sumWordsCount / sum_pc / vc) * pc / sum_pc
          (ww, div)
        }.reduceByKey(_ + _) //.sortBy(_._2, false)

        val signature = kld.join(wordsCount)
                           .map {case (w, (div, count)) => (w, div * count)}
                           // .map {case (w, (div, count)) => ((w, div, count), div * count)}
                          .filter(_._2 >= 400000.0)
                          .map(_._1)
                          // .sortBy(_._2, false)
                          .collect

        val sharedSignature = sc.broadcast(signature)

        val dictList = stat.flatMap(_._1.productIterator.map(_.asInstanceOf[String]))
                           .distinct.collect
        val dictSize = dictList.size
        val dictMap = dictList.zipWithIndex.toMap
        val sharedDictMap = sc.broadcast(dictMap)

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
        // val jsd = pairWise.filter {case (a, b) => a._1 == "doubt" && b._1 == "science"}
        //                   .map {case (vector1, vector2) =>
          (vector1._1, vector2._1,
            computeJSDivergence(vector1._2.asInstanceOf[SparseVector],
              vector2._2.asInstanceOf[SparseVector]))}.cache

        // end of program
        val jsd_local = jsd.collect()
        jsd_local.foreach(println)
        println
        sparseVectors.join(wordsCount).collect().foreach(println)
        println
        dictList.foreach(println)

        // var remainingPoints = sparseVectors.map(_._1).collect()
        // println("Remain: " + remainingPoints.size.toString)
        // val t1 = 0.25
        // val t2 = 0.25
        // while (remainingPoints.nonEmpty) {
        //   val sample = remainingPoints(Random.nextInt(remainingPoints.size))
        //   val dist = jsd_local.filter {case (w1, w2, c) => (w1 == sample || w2 == sample)}
        //                       //.collect
        //   val entropy = dist.filter(_._3 <= t1)
        //                     .flatMap(p => Array(p._1, p._2))
        //                     .filter(remainingPoints.contains(_))
        //                     .distinct
        //   println(sample + " -> [" + entropy.mkString(", ") + "]")
        //   val closePoints = dist.filter(_._3 <= t2)
        //                         .flatMap(p => Array(p._1, p._2))
        //                         .distinct :+ sample
        //   remainingPoints = remainingPoints.filterNot(p => closePoints.contains(p))
        //   println("Remain: " + remainingPoints.size.toString)
        // }

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

