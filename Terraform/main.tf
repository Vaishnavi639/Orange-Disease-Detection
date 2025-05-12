provider "aws" {
  region = "ap-south-1" 
}

# ECS Cluster
resource "" "my_cluster" {
  name = "my-simple-cluster"
}

# Task Definition
resource "aws_ecs_task_definition" "my_task" {
  family                   = "my-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1
  memory                   = 512

  container_definitions = jsonencode([{
    name  = "orange-disease-detection"
    image = "vaishnavi639/orange-disease-detection:5ae972e1c8c394d0fc34ebaa0bd15b6701ef2b77"
    portMappings = [{
      containerPort = 5000
      hostPort      = 5000
    }]
  }])
}

# Security Group
resource "aws_security_group" "ecs_sg" {
  name        = "ecs-security-group"
  description = "Allow HTTP inbound traffic"

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    
  }

   ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    
  }

   ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    
  }

   ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    
  }




  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ECS Service
resource "aws_ecs_service" "my_service" {
  name            = "orange-disease-detection"
  cluster         = aws_ecs_cluster.my_cluster.id
  task_definition = aws_ecs_task_definition.my_task.arn
  launch_type     = "FARGATE"
  desired_count   = 1

  network_configuration {
    subnets          = ["subnet-12345678"]  
    assign_public_ip = true
    security_groups  = [aws_security_group.ecs_sg.id]
  }
}
