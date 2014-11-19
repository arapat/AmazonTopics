
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

object AmazonWordCount {
    
    def main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("AmazonWordCount")
                                       .set("spark.executor.memory", "10g")
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
        val counters = tokens.map((_, 1))
                             .reduceByKey(_ + _)
        val candidates = counters.filter(_._2 >= 10)
                                 .filter(_._2 <= (reviewsCount * 0.1).toInt)
                                 .sortBy(_._2, false)
        candidates.take(100).foreach(println)

        println
        println("The number of reviews: " + reviewsCount.toString)
        println("Total tokens: " + tokens.count.toString)
        println("Total unique tokens: " + counters.count.toString)
        println("The number of unique tokens appear at least 10 times and at most in 10% of documents")
        println(candidates.count)

        sc.stop()
    }
}

