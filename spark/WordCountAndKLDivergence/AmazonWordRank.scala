
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

object AmazonWordRank {
    val unit = 0.0001
    
    def main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("AmazonKLDivergence")
                                       .set("spark.executor.memory", "50g")
        val sc = new SparkContext(sparkConf)
        // val lines = sc.textFile("/user/arapat/amazon/Electronics.txt")
        val lines = sc.textFile("/user/arapat/amazon/all.txt")

        val reviews = lines.map(_.split(":")
                           .map(_.trim))
                           .filter(_(0) == "review/text")
                           .map(_(1))
        val reviewsCount = reviews.count
        val tokens = reviews.flatMap(line =>
            "[a-zA-Z]+".r findAllIn line map (_.toLowerCase))
        val wordsCount = tokens.map((_, unit.toDouble))
                               .reduceByKey(_ + _)
                               .cache()

        val pairsCount = reviews.map(line =>
              "[a-zA-Z]+".r findAllIn line map (_.toLowerCase))
            .flatMap(array =>
              for { a <- array; b <- array if (a != b) } yield (a, b))
            .map((_, unit.toDouble))
            .reduceByKey(_ + _)
        val pairsSum = pairsCount.map(pair => (pair._1._1, pair._2))
                                 .reduceByKey(_ + _)

        val total = wordsCount.map(_._2).reduce(_ + _)
        val stat = pairsCount.map(a => (a._1._1, a))
                             .join(wordsCount)
                             .filter(_._2._2 >= 10 * unit)
                             .map(a => (a._1, a._2._1))
                             .join(pairsSum)
                             .map {case (w, (((ww, vv), pc), wps)) =>
                               (vv, (ww, wps, pc))}
                             .join(wordsCount)
                             .map {case (v, ((ww, wps, pc), vc)) =>
                               (ww, (wps, vc, pc, total))}

        val kld = stat.map { case (w, (wps, vc, pc, tc)) =>
          val div =
            math.log(pc * tc / wps / vc) * pc / wps
          (w, div)
        }.reduceByKey(_ + _) //.sortBy(_._2, false)

        val rank = kld.join(wordsCount)
                      .map {case (w, (div, count)) =>
                        (w, (div * count, div, count))}
                      .sortBy(_._2._1, false)
                      .cache()

        rank.collect().foreach(println)

        println
        println("Total tokens: " + rank.count)

        sc.stop()
    }
}

