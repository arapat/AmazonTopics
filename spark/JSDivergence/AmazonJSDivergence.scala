
import math.log

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

object AmazonJSDivergence {
    
    val lnOf2 = log(2)

    def computeJSDivergence(item: ((String, Array[(String, Int)]), (String, Array[(String, Int)]))) = {
        val itemA = item._1
        val itemB = item._2
        val a = itemA._1
        val vecA = itemA._2
        val b = itemB._1
        val vecB = itemB._2

        val sumA = vecA.map(_._2).sum.toDouble
        val sumB = vecB.map(_._2).sum.toDouble
        var i = 0
        var j = 0
        var result = 0.0

        while (i < vecA.size && j < vecB.size) {
          if (vecA(i)._1 < vecB(j)._1) {
            result = result + vecA(i)._2 / sumA
            i = i + 1
          } else if (vecA(i)._1 > vecB(j)._1) {
            result = result + vecB(j)._2 / sumB
            j = j + 1
          } else { // same word
            val a = vecA(i)._2 / sumA
            val b = vecB(j)._2 / sumB
            val m = 0.5 * (a + b)
            result = result + (a * log(a / m) / lnOf2 + b * log(b / m) / lnOf2)
            i = i + 1
            j = j + 1
          }
        }

        result = result + vecA.slice(i, vecA.size).map(_._2).sum / sumA
        result = result + vecB.slice(j, vecB.size).map(_._2).sum / sumB
        
        (a, b, result / 2.0)
      }

    def main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("AmazonJSDivergence")
                                       .set("spark.executor.memory", "50g")
        val sc = new SparkContext(sparkConf)
        val unit = 1 // or 0.0001.toDouble if the sum is too large to fit in Integer
        // val lines = sc.textFile("/user/arapat/amazon/Electronics.txt")
        val lines = sc.textFile("/user/arapat/amazon/all.txt")
        // or sc.textFile("/user/arapat/amazon/Electronics.txt") for small dataset

        // Format reviews
        val reviews = lines.map(_.split(":")
                           .map(_.trim))
                           .filter(_(0) == "review/text")
                           .map(_(1))
        val reviewsCount = reviews.count

        // extract tokens
        val tokens = reviews.flatMap(line =>
            "[a-zA-Z]+".r findAllIn line map (_.toLowerCase))
        val wordsCount = tokens.map((_, unit))
                             .reduceByKey(_ + _)
        // val tokensId = wordsCount.sortBy { case (w, count) => w }
        //                          .collect.zipWithIndex
        //                          .map { case ((w, count), index) => (w, (index, count)) }.toMap

        // generate token pairs
        val pairsCount = reviews.map(line =>
              "[a-zA-Z]+".r findAllIn line map (_.toLowerCase))
            .flatMap(array =>
                for { a <- array; b <- array if (a != b) } yield (a, b))
            .map((_, unit))
            .reduceByKey(_ + _)
            .map { case ((w, v), pc) => (w, ((w, v), pc)) }
            .join(wordsCount)
            .filter(_._2._2 >= 10) 
            .map { case (w, (((ww, vv), pc), wc)) => (vv, ((ww, vv), pc))}
            .join(wordsCount)
            .filter(_._2._2 >= 10)
            .map { case (v, (((ww, vv), pc), vc)) => ((ww, vv), pc)}

        val vector = pairsCount.map(pair => (pair._1._1, Array((pair._1._2, pair._2))))
                               .reduceByKey(_ ++ _)
                               .map { case (a, b) => (a, b.sortBy(k => k._1)) }
                               .persist()

        val sampleFraction = 0.5
        for (i <- 0 to 2000) {
          println("Sample " + (i+1).toString)
          // val sample = vector.sample(false, sampleFraction)
          val sample = vector.takeSample(false, 1)(0)
          // val cartesian = sample.cartesian(sample)
          //                       .filter { case ((a, vecA), (b, vecB)) => a < b } 
          // val jsd = cartesian.map(computeJSDivergence)
          //                    .collect()
          //                    .foreach(println)
          // println(computeJSDivergence((sample(0), sample(1))))
          val func = (data: (String, Array[(String, Int)])) => computeJSDivergence((sample, data))
          println(vector.sample(false, sampleFraction)
                        .map(func)
                        .collect()
                        .mkString("\n"))
        }

        println("Done.")


        // println
        // println("The number of reviews: " + reviewsCount.toString)
        // println("Total tokens: " + tokens.count.toString)
        // println("The number of unique tokens: " + wordsCount.count.toString)
        // println("The number of token pairs: " + pairsCount.count.toString)
        // println("The size of kld: " + kld.count.toString)

        sc.stop()
    }
}

