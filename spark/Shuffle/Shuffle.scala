
import math.log

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

object Shuffle {
    
    def main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("AmazonJSDivergence")
                                       .set("spark.executor.memory", "50g")
        val sc = new SparkContext(sparkConf)
        val lines = sc.textFile("/user/arapat/amazon/jsd.txt")
        val seed = 1000
        lines.takeSample(false, 20000, seed).foreach(println)
        sc.stop()
    }
}

