package org.booking.recommended

import org.bytedeco.javacpp._
import org.bytedeco.tritonserver.tritonserver._
import org.bytedeco.tritonserver.global.tritonserver._

import java.nio.file.Path
import java.util.UUID.randomUUID
import scala.collection.mutable
import scala.concurrent.{Await, Future, Promise}
import scala.concurrent.duration.{Duration, DurationInt, DurationLong, FiniteDuration, NANOSECONDS}
import scala.io.Source
import scala.util.Success
import scala.util.control.Breaks.{break, breakable}


case class RequestSingle(params: Map[String, Seq[String]]){
  override def toString: String = {
    params.take(4).map(kv => s"${kv._1}=${kv._2.head}").mkString(", ")
  }
}

case class InferenceResponse(promise: Promise[Float], request: RequestSingle, startTime: Long = System.nanoTime())

abstract class TritonModel {
  val modelName: String
  val modelVersion: String
  val modelRepoPath: Path
  var startTime: Long = System.nanoTime()

  val live: Array[Boolean] = Array(false)
  def isLive: Boolean =  live(0)
  val ready: Array[Boolean] = Array(false)
  def isReady: Boolean = ready(0)
  val requested_memory_type = TRITONSERVER_MEMORY_CPU;
  val responseAlloc = new ResponseAlloc();
  val responseRelease = new ResponseRelease();
  val inferRequestComplete = new InferRequestComplete();
  val inferResponseComplete = new InferResponseComplete();

  val server = new TRITONSERVER_Server(null);
  val allocator = new TRITONSERVER_ResponseAllocator(null)


  startTime = System.nanoTime()
  val serverOptions = new TRITONSERVER_ServerOptions(null)
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsNew(serverOptions), "creating server options")
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetModelRepositoryPath(serverOptions, modelRepoPath.toString), "Setting model repo path")
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogVerbose(serverOptions, 0), "Setting verbose logging level")
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetBackendDirectory(serverOptions, "/opt/tritonserver/backends"), "setting backend directory")
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetRepoAgentDirectory(serverOptions, "/opt/tritonserver/repoagents"), "setting repository agent directory")
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetStrictModelConfig(serverOptions, true), "setting strict model configuration")
  //  Initialise Server
  FAIL_IF_ERR(TRITONSERVER_ServerNew(server, serverOptions), "creating server")
  FAIL_IF_ERR(TRITONSERVER_ServerOptionsDelete(serverOptions), "deleting server options")
  FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorNew(allocator, responseAlloc, responseRelease, null), "creating response allocator")
  println(s"Starting the server took ${(System.nanoTime()-startTime).nano.toMillis}ms")


  val futures = new mutable.HashMap[Pointer, InferenceResponse]()

  class ResponseAlloc extends TRITONSERVER_ResponseAllocatorAllocFn_t {
    override def call(allocator: TRITONSERVER_ResponseAllocator, tensor_name: String,
                      byte_size: Long, memory_type: Int, memory_type_id: Long, userp: Pointer,
                      buffer: PointerPointer[_ <: Pointer], buffer_userp: PointerPointer[_ <: Pointer],
                      actual_memory_type: IntPointer, actual_memory_type_id: LongPointer): TRITONSERVER_Error = {
      // Initially attempt to make the actual memory type and id that we
      // allocate be the same as preferred memory type
      actual_memory_type.put(0L, memory_type)
      actual_memory_type_id.put(0L, memory_type_id)

      // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
      // need to do any other book-keeping.
      if(byte_size == 0){
        buffer.put(0,null)
        buffer_userp.put(0,null)
        //        println(s"allocated $byte_size bytes for result tensor $tensor_name")
      } else {
        var allocatedPointer = new Pointer()
        actual_memory_type.put(0L, requested_memory_type)
        actual_memory_type.put(0L, TRITONSERVER_MEMORY_CPU)

        allocatedPointer = Pointer.malloc(byte_size)
        // Pass the tensor name with buffer_userp so we can show it when
        // releasing the buffer.
        if (!allocatedPointer.isNull){
          buffer.put(0, allocatedPointer)
          buffer_userp.put(0, Loader.newGlobalRef(tensor_name))
          //          println(s"allocated $byte_size bytes in ${TRITONSERVER_MemoryTypeString(actual_memory_type.get())} for result tensor $tensor_name")
        }
      }
      null
    }
  }

  class ResponseRelease extends TRITONSERVER_ResponseAllocatorReleaseFn_t {
    override def call(allocator: TRITONSERVER_ResponseAllocator, buffer: Pointer, buffer_userp: Pointer,
                      byte_size: Long, memory_type: Int, memory_type_id: Long): TRITONSERVER_Error = {
      var name = ""
      if (buffer_userp != null){
        name = Loader.accessGlobalRef(buffer_userp).toString
      } else {
        name = "<unknown>"
      }

      //      println(s"Releasing buffer $buffer of size $byte_size in ${TRITONSERVER_MemoryTypeString(memory_type)} for result '$name'")
      Pointer.free(buffer)
      Loader.deleteGlobalRef(buffer_userp)

      null
    }
  }

  class InferRequestComplete extends TRITONSERVER_InferenceRequestReleaseFn_t {
    override def call(request: TRITONSERVER_InferenceRequest, flags: Int, userp: Pointer): Unit = {
//      FAIL_IF_ERR(TRITONSERVER_InferenceRequestDelete(request), "deleting inference request")
    }
  }

  def extractOutputFromResponse(response: TRITONSERVER_InferenceResponse): Float = {
    val cname = new BytePointer()
    val datatype = new IntPointer(1L)
    val shape = new LongPointer()
    val dim_count = new LongPointer(1)
    val base = new FloatPointer(4)
    val byte_size = new SizeTPointer(1)
    val memory_type = new IntPointer(1L)
    val memory_type_id = new LongPointer(1)
    val userp = new Pointer()
    val outputCount = new IntPointer(1L)
    FAIL_IF_ERR(TRITONSERVER_InferenceResponseOutputCount(response, outputCount), "getting number of response outputs")

    FAIL_IF_ERR(
      TRITONSERVER_InferenceResponseOutput(
        response, 0, cname, datatype, shape, dim_count, base,
        byte_size, memory_type, memory_type_id, userp),
      "getting output info")
//    val odata = new BytePointer(byte_size.get())
//    odata.put(base.limit(byte_size.get()))
    println(s"Count: ${outputCount.get()}, Byte Size: ${byte_size.get()}, Base: ${base.get()}, Cname: ${cname.getString()} dtype: ${datatype.get()}, shape: ${shape.get()}")
//    base.limit(byte_size.get()).get()
    base.get()
  }

  class InferResponseComplete extends TRITONSERVER_InferenceResponseCompleteFn_t {
    override def call(response: TRITONSERVER_InferenceResponse, flags: Int, userp: Pointer): Unit = {
      if (response!=null){
        val probability = extractOutputFromResponse(response)
        val inferenceResponse = futures(userp)
        inferenceResponse.promise.tryComplete(Success(probability))
        println(s"Forward pass took: ${Duration(System.nanoTime() - inferenceResponse.startTime, NANOSECONDS).toMillis}ms")
//        FAIL_IF_ERR(TRITONSERVER_InferenceResponseDelete(response), "deleting inference response")
        futures.remove(userp)
      }
    }
  }

  def FAIL(message: String): Unit = {
    println("Failure: " + message)
    System.exit(1)
  }

  def FAIL_IF_ERR(err: TRITONSERVER_Error, message: String): Unit = {
    if (err != null) {
      println("error: " + message + ":"
        + TRITONSERVER_ErrorCodeString(err) + " - "
        + TRITONSERVER_ErrorMessage(err))
      TRITONSERVER_ErrorDelete(err);
      System.exit(1);
    }
  }

  def waitTillReady(server: TRITONSERVER_Server,timeout: FiniteDuration = 10.seconds): Unit = {
    val startTime = System.nanoTime()
    breakable {
      while (true) {
        healthCheck(server)
        if (isLive && isReady) break
        if (Duration(System.nanoTime()-startTime, NANOSECONDS).toSeconds > timeout.toSeconds) {
          FAIL("Timed out trying to find a healthy and live server")
        }
      }
    }
  }

  def healthCheck(server: TRITONSERVER_Server): Unit = {
    FAIL_IF_ERR(TRITONSERVER_ServerIsLive(server, live),"unable to get server liveness")
    FAIL_IF_ERR(TRITONSERVER_ServerIsReady(server, ready), "unable to get server readiness")
    FAIL_IF_ERR(TRITONSERVER_ServerModelIsReady(server, modelName, modelVersion.toLong, ready), "unable to get model readiness")

  }

  def printServerMetadata(server: TRITONSERVER_Server): Unit = {
    val serverMetadataMessage = new TRITONSERVER_Message()
    FAIL_IF_ERR(TRITONSERVER_ServerMetadata(server, serverMetadataMessage), "unable to get server metadata message")
    val buffer = new BytePointer()
    val byteSize = new SizeTPointer(1)
    FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(serverMetadataMessage, buffer, byteSize), "unable to serialize server metadata message")
    println("Server Status:")
    println(buffer.limit(byteSize.get).getString)
    FAIL_IF_ERR(TRITONSERVER_MessageDelete(serverMetadataMessage), "deleting status metadata")
  }

  def printModelMetadata(server: TRITONSERVER_Server): Unit = {
    val modelMetadataMessage = new TRITONSERVER_Message()
    FAIL_IF_ERR(TRITONSERVER_ServerModelMetadata(server,modelName,modelVersion.toLong, modelMetadataMessage), "unable to get model metadata message")
    val buffer = new BytePointer()
    val byteSize = new SizeTPointer(1)
    FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(modelMetadataMessage, buffer, byteSize), "unable to serialize model status protobuf")
    println("Model Status:")
    println(buffer.limit(byteSize.get).getString)
    FAIL_IF_ERR(TRITONSERVER_MessageDelete(modelMetadataMessage), "deleting status metadata")
  }

  def getInferenceRequestObj(reqId: String): TRITONSERVER_InferenceRequest

  def predictSync(request: RequestSingle): Float

  def supportedFeatures: Seq[String]

  def getOutputSpaceSize: Int

  def shutdown(): Unit = {
    FAIL_IF_ERR(TRITONSERVER_ResponseAllocatorDelete(allocator), "deleting response allocator")
    FAIL_IF_ERR(TRITONSERVER_ServerDelete(server), "deleting the server")
  }
}

class TritonLightGbm(val modelName: String, val modelVersion: String, val modelRepoPath: Path) extends TritonModel {

  val modelFile = modelRepoPath.resolve(s"$modelName/$modelVersion/model.txt")
  val modelConfigPath = modelRepoPath.resolve(s"$modelName/config.pbtxt")
  val modelString = getModelString

  val supportedFeatures: Seq[String] = {
    modelString.split("\n").filter(_.startsWith("feature_names=")).head.split("=")(1).split(" ")
  }

  waitTillReady(server)
  printServerMetadata(server)
  printModelMetadata(server)

  val inputTensorName = "input__0"
  val outputTensorName = "output__0"
  val inputData = Array(new FloatPointer(supportedFeatures.length.toLong))
  val inputDataPointer = inputData(0).getPointer(classOf[BytePointer])
  val inputDataSize = inputDataPointer.limit()
  val inputDataBase = inputDataPointer


  def getInferenceRequestObj(reqId: String): TRITONSERVER_InferenceRequest = {
    val irequest = new TRITONSERVER_InferenceRequest()
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestNew(irequest, server, modelName, modelVersion.toLong), "creating inference request")
    // TODO: Find better id
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetId(irequest, reqId), "setting ID for the request")

    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetReleaseCallback(irequest, inferRequestComplete, null), "setting request release callback")

    val inputShape = Array(1L, supportedFeatures.length.toLong)
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(irequest, inputTensorName, TRITONSERVER_TYPE_FP32, inputShape,inputShape.length), s"setting input $inputTensorName meta-data for the request")

    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, outputTensorName), "requesting output probability for the request")
    irequest

  }

  def addInputData(inputRow: Seq[Float]): Unit = {
    for (i <- supportedFeatures.indices){
      inputData(0).put(i,inputRow(i))
    }
  }

  def getModelString: String = {
    val src = Source.fromFile(modelFile.toString)
    try {
      src.getLines().mkString("\n")
    } finally {
      src.close()
    }
  }

  def makePredictions(request: RequestSingle, timeoutNano: Long): Future[Float] = {
    val retPromise =  Promise[Float]()


    startTime = System.nanoTime()
    val irequest = getInferenceRequestObj(randomUUID().toString)
    println(s"Creating request object took ${(System.nanoTime()-startTime).nano.toMillis}ms")

    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(irequest, inputTensorName,
      inputDataBase,inputDataSize,requested_memory_type,0),s"assigning $inputTensorName data")

    startTime = System.nanoTime()
    val inputRow = supportedFeatures.map(
      featName => try {
        val values = request.params.get(featName)
        if (values.isEmpty || values.get.isEmpty || values.get.head == null) {
          Float.NaN
        } else {
          values.get.head.toFloat
        }
      } catch {
        case e: Exception =>
          throw new IllegalArgumentException(s"Invalid value '${request.params.get(featName).map(_.head)}' for feature '$featName': ${e.getClass.getSimpleName}")
      }
    )
    addInputData(inputRow)
    println(s"Populating input data took ${(System.nanoTime()-startTime).nano.toMillis}ms")

    val inferenceResponse = InferenceResponse(retPromise,request)

    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(irequest, allocator, null, inferResponseComplete, irequest), "setting response callback")
    futures.put(irequest, inferenceResponse)
    FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(server, irequest, null /* trace */),"running inference")
    retPromise.future
  }


  def predictSync(request: RequestSingle): Float = {
    val timeout = 5000.millis
    Await.result(makePredictions(request,timeout.toNanos),timeout)
  }



  def getOutputSpaceSize: Int = {
    modelString.split("\n").filter(_.startsWith("num_class=")).head.split("=")(1).toInt
  }

}

class TritonTF(val modelName: String, val modelVersion: String, val modelRepoPath: Path) extends TritonModel {

  waitTillReady(server)
  printServerMetadata(server)
  printModelMetadata(server)

  val input1TensorName = "input_1"
  val input2TensorName = "input_2"
  val outputTensorName = "output_1"
  val inputDtype = TRITONSERVER_TYPE_INT64



  def getInferenceRequestObj(reqId: String): TRITONSERVER_InferenceRequest = {
    val irequest = new TRITONSERVER_InferenceRequest()
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestNew(irequest, server, modelName, modelVersion.toLong), "creating inference request")
    // TODO: Find better id
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetId(irequest, reqId), "setting ID for the request")

    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetReleaseCallback(irequest, inferRequestComplete, null), "setting request release callback")

    val inputShape = Array(1L, 1L)
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(irequest, input1TensorName, inputDtype, inputShape,inputShape.length), s"setting input $input1TensorName meta-data for the request")
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddInput(irequest, input2TensorName, inputDtype, inputShape,inputShape.length), s"setting input $input2TensorName meta-data for the request")
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, outputTensorName), "requesting output probability for the request")
    irequest

  }

//  def addInputData(inputRow: Seq[Int]): Unit = {
//    input1Data(0).put(inputRow(0))
//    input2Data(0).put(inputRow(1))
//  }

  def makePredictions(request: RequestSingle, timeoutNano: Long): Future[Float] = {
    val retPromise =  Promise[Float]()
    startTime = System.nanoTime()
    val irequest = getInferenceRequestObj(randomUUID().toString)
    println(s"Creating request object took ${(System.nanoTime()-startTime).nano.toMillis}ms")

    val input1Data = Array(new IntPointer(1L))
    val input2Data = Array(new IntPointer(1L))
    val input1DataPointer = input1Data(0).getPointer(classOf[BytePointer])
    val input2DataPointer = input2Data(0).getPointer(classOf[BytePointer])
    val inputDataSize = input1DataPointer.limit()
    val input1DataBase = input1DataPointer
    val input2DataBase = input2DataPointer



    startTime = System.nanoTime()
    val inputRow = supportedFeatures.map(
      featName => try {
        val values = request.params.get(featName)
        if (values.isEmpty || values.get.isEmpty || values.get.head == null) {
          42
        } else {
          values.get.head.toInt
        }
      } catch {
        case e: Exception =>
          throw new IllegalArgumentException(s"Invalid value '${request.params.get(featName).map(_.head)}' for feature '$featName': ${e.getClass.getSimpleName}")
      }
    )
//    addInputData(inputRow)
    input1Data(0).put(inputRow(0))
    input2Data(0).put(inputRow(1))

    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(irequest, input1TensorName,
      input1DataBase,inputDataSize,requested_memory_type,0),s"assigning $input1TensorName data")
    FAIL_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(irequest, input2TensorName,
      input2DataBase,inputDataSize,requested_memory_type,0),s"assigning $input2TensorName data")
    println(s"Populating input data took ${(System.nanoTime()-startTime).nano.toMillis}ms")

    val inferenceResponse = InferenceResponse(retPromise,request)

    FAIL_IF_ERR(TRITONSERVER_InferenceRequestSetResponseCallback(irequest, allocator, null, inferResponseComplete, irequest), "setting response callback")
    futures.put(irequest, inferenceResponse)
    FAIL_IF_ERR(TRITONSERVER_ServerInferAsync(server, irequest, null /* trace */),"running inference")

    retPromise.future
  }


  def predictSync(request: RequestSingle): Float = {
    val timeout = 5000.millis
    Await.result(makePredictions(request,timeout.toNanos),timeout)
  }

  def supportedFeatures: Seq[String] = {
    Seq("input_1","input_2")
  }

  override def getOutputSpaceSize: Int = 1
}

object Debug extends App {
  println(s"Running Triton Test")
  var startTime = System.nanoTime()
  if (args.length == 0) {
    println("You need to give the model to run. ")
    System.exit(0)
  }
  val modelFormat = args(0)
  if (modelFormat != "tf" && modelFormat != "lgbm"){
    println(s"Invalid model format $modelFormat, Use 'tf' or 'lgbm'")
    System.exit(0)
  }
  val modelDir = Path.of("/var/lib/recommended/models/")
  val modelName = modelFormat match {
    case "tf" => "sample_tf"
    case "lgbm" => "sample_lightgbm"
  }
  val modelVersion = "1"
  val model = modelName match {
    case "sample_tf" => new TritonTF(modelName = modelName, modelVersion = modelVersion, modelRepoPath = modelDir)
    case "sample_lightgbm" => new TritonLightGbm(modelName = modelName, modelVersion = modelVersion, modelRepoPath = modelDir)
  }

  val rand = new scala.util.Random

  startTime = System.nanoTime()
  val testCases = (0 until 100).map(i => {
    RequestSingle(model.supportedFeatures.map(featureName => {
      featureName -> Seq(rand.nextInt(10).toString)
    }).toMap)
  })
  println(s"Creating random test data took ${(System.nanoTime()-startTime).nano.toMillis}ms")

  testCases.foreach(request => {
    startTime = System.nanoTime()
    val result = model.predictSync(request)
    println(s"model.predictSync took ${(System.nanoTime()-startTime).nano.toMillis}ms")
    println(s"\n------Got prediction $result for input ${request.toString}..")
  })

  println("Reached the end!!")



}
