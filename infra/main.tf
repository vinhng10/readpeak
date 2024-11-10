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
resource "aws_kinesis_stream" "kinesis_stream" {
  name        = "KinesisStream"
  shard_count = 1

  stream_mode_details {
    stream_mode = "PROVISIONED"
  }

  tags = {
    Environment = "dev"
  }
}
