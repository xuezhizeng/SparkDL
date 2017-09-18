package org.apache.spark.ml.image.core

object BGRImageNormalizer {
  def apply(mean: Array[Float], std: Array[Float]): BGRImageNormalizer =
    new BGRImageNormalizer(mean, std)
}


class BGRImageNormalizer (mean: Array[Float], std: Array[Float])
  extends TransformStep[(Array[Float], Array[Int]), (Array[Float], Array[Int])] {

  override def apply(prev: (Array[Float], Array[Int])): (Array[Float], Array[Int]) = {
    val content = prev._1
    val dimension = prev._2
    require(content.length == dimension.product)

    val meanR = mean(0)
    val meanG = mean(1)
    val meanB = mean(2)

    val stdR = std(0)
    val stdG = std(1)
    val stdB = std(2)

    require(content.length % 3 == 0)
    var i = 0
    while (i < content.length) {
      content(i + 2) = ((content(i + 2) - meanR) / stdR).toFloat
      content(i + 1) = ((content(i + 1) - meanG) / stdG).toFloat
      content(i + 0) = ((content(i + 0) - meanB) / stdB).toFloat
      i += 3
    }
    (content, dimension)
  }

}
