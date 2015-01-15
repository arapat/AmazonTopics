
import math.log
import math.max
import math.min
import util.Random

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{Vector, Vectors, SparseVector}

object DistributionalSimilarity {
    val stopwords = Array("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "ve", "ll", "haven't", "hasn't", "don't", "doesn't")

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
        val lines = sc.textFile("/user/arapat/dist-sim/train")

        val sharedStopwords = sc.broadcast(stopwords)
        val pairsCount = lines.map(_.split('\t'))
                              .map (p => ((p(1).trim, p(2).trim), p(0).trim.toDouble))
                              .cache
        val pairsSum = pairsCount.map(p => (p._1._1, p._2))
                                 .reduceByKey(_ + _)

        // Compute the product of KL divergence and word counts 
        // for each words that appear more than 10 times
        val stat = pairsCount.map(p => (p._1._1, p))
                             .join(pairsSum)
                             .map { case (w, (((ww, vv), pc), sum_pc)) =>
                              ((ww, vv), (sum_pc, pc)) }
                             .cache

        val dictList = stat.map(_._1._2)
                           .distinct.collect

        val dictSize = dictList.size
        val dictMap = dictList.zipWithIndex.toMap
        val sharedDictMap = sc.broadcast(dictMap)

        val preVector = stat.map {case ((ww, v), (sum_pc, pc)) =>
                                    (ww, (Array(sharedDictMap.value(v)), Array(pc / sum_pc)))}
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

        println(sparseVectors.count)
        println(dictSize)
        println

        val jsd_local = jsd.collect()
        jsd_local.foreach(println)
        println
        sparseVectors.join(pairsSum).collect().foreach(println)
        println
        dictList.foreach(println)
        println
        println("Done.")
        return

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

