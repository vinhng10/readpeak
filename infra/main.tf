terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region = "us-east-1"
}

# Access the effective Account ID
data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
}

resource "aws_s3_bucket" "code_bucket" {
  bucket = "ads-code"

  tags = {
    Environment = "dev"
  }
}

resource "aws_s3_bucket" "datalake_bucket" {
  bucket = "ads-datalake"

  tags = {
    Environment = "dev"
  }
}

# resource "aws_s3_object" "kinesis_analytics_code" {
#   bucket = aws_s3_bucket.code_bucket.id
#   key    = "kinesis-analytics.jar"
#   source = "kinesis-analytics.jar"

#   tags = {
#     Environment = "dev"
#   }
# }

resource "aws_kinesis_stream" "kinesis_stream" {
  name        = "KinesisStream"
  shard_count = var.shard_count

  shard_level_metrics = [
    "IncomingBytes",
    "OutgoingBytes",
  ]

  stream_mode_details {
    stream_mode = "PROVISIONED"
  }

  tags = {
    Environment = "dev"
  }
}

# resource "aws_kinesisanalyticsv2_application" "kinesis_analytics" {
#   name                   = "KinesisAnalytics"
#   runtime_environment    = "FLINK-1_19"
#   service_execution_role = aws_iam_role.kinesis_analytics_role.arn

#   application_configuration {
#     application_code_configuration {
#       code_content {
#         s3_content_location {
#           bucket_arn = aws_s3_bucket.code_bucket.arn
#           file_key   = aws_s3_object.kinesis_analytics_code.key
#         }
#       }

#       code_content_type = "ZIPFILE"
#     }
#   }

#   tags = {
#     Environment = "dev"
#   }
# }

# resource "aws_iam_role" "kinesis_analytics_role" {
#   name = "KinesisAnalyticsRole"

#   assume_role_policy = jsonencode({
#     Version = "2012-10-17"
#     Statement = [
#       {
#         Action = "sts:AssumeRole"
#         Effect = "Allow"
#         Principal = {
#           Service = "kinesisanalytics.amazonaws.com"
#         },
#         Condition = {
#           StringEquals = {
#             "aws:SourceAccount" = "${account_id}"
#           },
#           ArnEquals = {
#             "aws:SourceArn" = aws_kinesisanalyticsv2_application.kinesis_analytics.arn
#           }
#         }
#       }
#     ]
#   })

#   tags = {
#     Environment = "dev"
#   }
# }

# resource "aws_iam_policy" "kinesis_analytics_policy" {
#   name        = "KinesisAnalyticsDataStreamAndS3Access"
#   description = "Policy to allow full access to Kinesis Data Streams and S3 for Kinesis Analytics"

#   policy = jsonencode({
#     Version = "2012-10-17"
#     Statement = [
#       {
#         Sid    = "AccessKinesisStream",
#         Effect = "Allow",
#         Action = [
#           "kinesis:DescribeStream",
#           "kinesis:GetRecords",
#           "kinesis:GetShardIterator",
#           "kinesis:ListStreams",
#           "kinesis:PutRecord",
#           "kinesis:PutRecords"
#         ],
#         Resource = aws_kinesis_stream.kinesis_stream.arn
#       },
#       {
#         Sid    = "ReadCode",
#         Effect = "Allow",
#         Action = [
#           "s3:GetObject",
#           "s3:GetObjectVersion"
#         ],
#         Resource = aws_s3_object.kinesis_analytics_code.arn
#       },
#       {
#         Sid    = "AccessDatalake",
#         Effect = "Allow",
#         Action = [
#           "s3:Abort*",
#           "s3:DeleteObject*",
#           "s3:GetObject*",
#           "s3:GetBucket*",
#           "s3:List*",
#           "s3:ListBucket",
#           "s3:PutObject"
#         ],
#         Resource = [
#           "${aws_s3_bucket.datalake_bucket.arn}",
#           "${aws_s3_bucket.datalake_bucket.arn}/*"
#         ]
#       }
#     ]
#   })

#   tags = {
#     Environment = "dev"
#   }
# }

# resource "aws_iam_role_policy_attachment" "kinesis_analytics_role_policy_attachment" {
#   policy_arn = aws_iam_policy.kinesis_analytics_policy.arn
#   role       = aws_iam_role.kinesis_analytics_role.name
# }
