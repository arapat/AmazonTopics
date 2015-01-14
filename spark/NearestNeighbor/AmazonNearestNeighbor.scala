
import math.log
import math.max
import math.min
import util.Random

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{Vector, Vectors, SparseVector}

object AmazonNearestNeighbor {
    val stopwords = Array("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "ve", "ll", "haven't", "hasn't", "don't", "doesn't", "i'm", "he's", "she's", "i've", "it's")

    // def computeJSDivergence(vec1: SparseVector, vec2: SparseVector) = {
    //   val indices = (vec1.indices ++ vec2.indices).distinct
    //   var result = 0.0
    //   for (i <- indices) {
    //     val a = vec1(i)
    //     val b = vec2(i)
    //     val m = (a + b) * 0.5
    //     if (a != 0) {
    //       result = result + a * log(a / m)
    //     }
    //     if (b != 0) {
    //       result = result + b * log(b / m)
    //     }
    //   }
    //   result / 2.0
    // }

    def computeJSDivergence(vec1: SparseVector, vec2: SparseVector) = {
      // var logs = Array((0, 0.0, 0.0))

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
          // logs :+= (0, p1(itr1)._2, result)
          itr1 = itr1 + 1
        } else if (p1(itr1)._1 > p2(itr2)._1) {
          result = result + p2(itr2)._2 * log(2)
          // logs :+= (1, p2(itr2)._2, result)
          itr2 = itr2 + 1
        } else {
          val m = (p1(itr1)._2 + p2(itr2)._2) / 2.0
          result = result + p1(itr1)._2 * log(p1(itr1)._2 / m) + p2(itr2)._2 * log(p2(itr2)._2 / m)
          // logs :+= (2, m, result)
          itr1 = itr1 + 1
          itr2 = itr2 + 1
        }
      }
      while (itr1 < len1) {
          result = result + p1(itr1)._2 * log(2)
          // logs :+= (0, p1(itr1)._2, result)
          itr1 = itr1 + 1
      }
      while (itr2 < len2) {
          result = result + p2(itr2)._2 * log(2)
          // logs :+= (1, p2(itr2)._2, result)
          itr2 = itr2 + 1
      }
      result / 2.0
      // (logs, p1, p2)
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
                               // .cache

        // word pair will appear twice: (w, v) and (v, w)
        // word w and v should each appear at least 10 times
        val pairsCount = reviews.map(line =>
              "[a-zA-Z]+".r findAllIn line map (_.toLowerCase))
            .flatMap(array =>
              for { a <- array; b <- array if (a != b) } yield (a, b))
            .filter {case (w1, w2) =>
              sharedStopwords.value.contains(w1) == false &&
              sharedStopwords.value.contains(w2) == false}
            .map(p => (p._1, p)).join(wordsCount).filter(_._2._2 >= 10)
            .map(p => (p._2._1._2, p._2._1)).join(wordsCount).filter(_._2._2 >= 10)
            .map(p => (p._2._1, 1.0))
            .reduceByKey(_ + _)
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

        val preVector = stat.map {case ((ww, v), (wps, vc, pc)) =>
                                    (ww, (Array(sharedDictMap.value(v)), Array(pc / wps)))}
        val sparseVectors = preVector.reduceByKey {
          (p1 : (Array[Int], Array[Double]), p2 : (Array[Int], Array[Double])) =>
            (p1._1 ++ p2._1, p1._2 ++ p2._2)
        }.map {case (ww, (index, value)) =>
                (ww, Vectors.sparse(dictSize, index, value))}.cache

        val centers = sparseVectors.filter {p =>
                              sharedSignature.value.contains(p._1)}.collect
        val sharedCenters = sc.broadcast(centers)

        val closest = sparseVectors.map {p =>
          (p._1, {for (w <- sharedCenters.value) yield
            (computeJSDivergence(w._2.asInstanceOf[SparseVector],
              p._2.asInstanceOf[SparseVector]), w._1)}.min)
        }.map { case (v, (dist, w)) => (w, (Array(v), dist)) }
         .reduceByKey {
           (p1 : (Array[String], Double), p2 : (Array[String], Double)) =>
             (p1._1 ++ p2._1, p1._2 + p2._2)}
         .map { case (w, (cluster, dist)) => (w, (cluster, dist / cluster.size))}

        closest.collect.foreach { case (w, (cluster, avg_dist)) =>
          println(w + " (" + avg_dist.toString + ")")
          println(cluster.mkString(", "))
          println
        }

        // println
        // println("Dict size: " + dictList.size)
        // println("Signature size: " + signature.size)

        // println(sparseVectors.count)
        // println(jsd.count)
        sc.stop()
        println("Done.")
    }
}

