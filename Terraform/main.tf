provider "aws" {
  region = "ap-south-1"  # Change to your preferred region
}

resource "aws_security_group" "orange_disease_sg" {
  name        = "orange_disease_sg"
  description = "Security group for Orange Disease Detection app"

  # Allow incoming traffic on port 5000 (app port)
  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Consider restricting this in production
  }

  # Allow incoming traffic on port 8000 (MLflow port)
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Consider restricting this in production
  }

  # Allow SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Consider restricting this in production
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "orange-disease-detection-sg"
  }
}

resource "aws_instance" "orange_disease_app" {
  ami           = "ami-0261755bbcb8c4a84"  # Amazon Linux 2 AMI, update as needed
  instance_type = "t2.micro"  # Adjust based on your model's requirements
  key_name      = "your-key-pair-name"  # Replace with your SSH key pair

  security_groups = [aws_security_group.orange_disease_sg.name]

  root_block_device {
    volume_size = 8  # GB, adjust as needed
    volume_type = "gp2"
  }

  user_data = <<-EOF
              #!/bin/bash
              sudo yum update -y
              sudo amazon-linux-extras install docker -y
              sudo service docker start
              sudo usermod -a -G docker ec2-user
              sudo systemctl enable docker
              sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
              sudo chmod +x /usr/local/bin/docker-compose
              # Docker will be configured by CI/CD
              EOF

  tags = {
    Name = "orange-disease-detection-app"
  }
}

output "instance_ip" {
  value = aws_instance.orange_disease_app.public_ip
}
