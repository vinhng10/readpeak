package com.example.flink;

import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.connectors.kinesis.FlinkKinesisConsumer;
import org.apache.flink.streaming.connectors.fs.bucketing.BucketingSink;
import org.apache.flink.streaming.connectors.fs.StringWriter;
import java.util.Properties;

public class KinesisToS3Job {

    public static void main(String[] args) throws Exception {
        // Setup the execution environment
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Configure AWS Kinesis consumer properties
        Properties kinesisConsumerConfig = new Properties();
        kinesisConsumerConfig.setProperty("aws.region", "us-west-2"); // Change to your region
        kinesisConsumerConfig.setProperty("aws.credentials.provider", "AUTO");
        kinesisConsumerConfig.setProperty("flink.stream.initpos", "LATEST"); // Start from the latest record

        String kinesisStreamName = "my-kinesis-stream"; // Replace with your Kinesis stream name

        // Create a Kinesis Data Stream source
        DataStream<String> kinesisStream = env.addSource(
                new FlinkKinesisConsumer<>(kinesisStreamName, new SimpleStringSchema(), kinesisConsumerConfig));

        // Process the data (Example: converting to uppercase)
        DataStream<String> processedStream = kinesisStream.map(String::toUpperCase);

        // Set up S3 sink
        BucketingSink<String> s3Sink = new BucketingSink<>("s3://my-bucket/output"); // Replace with your S3 bucket and path
        s3Sink.setBucketer(new DateTimeBucketer<>("yyyy-MM-dd--HHmm"));
        s3Sink.setWriter(new StringWriter<>());
        s3Sink.setBatchSize(1024 * 1024 * 400); // Optional: Set batch size (e.g., 400MB)

        // Add the sink to the pipeline
        processedStream.addSink(s3Sink);

        // Execute the Flink pipeline
        env.execute("Kinesis to S3 Flink Job");
    }
}
