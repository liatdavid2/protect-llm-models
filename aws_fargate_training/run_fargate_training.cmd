@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
set AWS_PAGER=

REM ===== Config =====
set AWS_REGION=us-east-1
set AWS_ACCOUNT_ID=127393435473

set ECR_REPO=secure-llm-guard-trainer
set LOCAL_IMAGE=secure-llm-guard-trainer:latest
set ECR_IMAGE=%AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPO%:latest

set ECS_CLUSTER=secure-llm-training-cluster
set TASK_FAMILY=secure-llm-guard-trainer-task
set CONTAINER_NAME=secure-llm-guard-trainer

set LOG_GROUP=/ecs/secure-llm-guard-trainer

set S3_BUCKET=secure-llm-gateway-models
set S3_PREFIX=secure-llm-gateway/guard-training/local-demo

set EXECUTION_ROLE_NAME=ecsTaskExecutionRole
set TASK_ROLE_NAME=secureLlmGuardTrainerTaskRole

echo.
echo ================================
echo  Secure LLM Guard Trainer - AWS
echo ================================
echo Region: %AWS_REGION%
echo ECR image: %ECR_IMAGE%
echo S3: s3://%S3_BUCKET%/%S3_PREFIX%
echo.

REM ===== Check AWS identity =====
echo [1/12] Checking AWS identity...
aws sts get-caller-identity
if errorlevel 1 goto error

REM ===== ECR login =====
echo.
echo [2/12] Logging in to ECR...
aws ecr get-login-password --region %AWS_REGION% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com
if errorlevel 1 goto error

REM ===== Ensure ECR repo exists =====
echo.
echo [3/12] Ensuring ECR repository exists...
aws ecr describe-repositories --repository-names %ECR_REPO% --region %AWS_REGION% >nul 2>&1
if errorlevel 1 (
    echo Creating ECR repository %ECR_REPO%...
    aws ecr create-repository --repository-name %ECR_REPO% --region %AWS_REGION%
    if errorlevel 1 goto error
) else (
    echo ECR repository already exists.
)

REM ===== Tag and push image =====
echo.
echo [4/12] Tagging Docker image...
docker tag %LOCAL_IMAGE% %ECR_IMAGE%
if errorlevel 1 goto error

echo.
echo [5/12] Pushing Docker image to ECR...
docker push %ECR_IMAGE%
if errorlevel 1 goto error

REM ===== Create ECS cluster =====
echo.
echo [6/12] Creating ECS cluster if needed...
aws ecs describe-clusters --clusters %ECS_CLUSTER% --region %AWS_REGION% --query "clusters[0].clusterName" --output text > ecs_cluster_check.txt 2>nul
set /p CLUSTER_CHECK=<ecs_cluster_check.txt
del ecs_cluster_check.txt

if "%CLUSTER_CHECK%"=="%ECS_CLUSTER%" (
    echo ECS cluster already exists.
) else (
    aws ecs create-cluster --cluster-name %ECS_CLUSTER% --region %AWS_REGION%
    if errorlevel 1 goto error
)

REM ===== Create log group =====
echo.
echo [7/12] Creating CloudWatch log group if needed...
aws logs create-log-group --log-group-name %LOG_GROUP% --region %AWS_REGION% >nul 2>&1
echo Log group ready.

REM ===== Create trust policy =====
echo.
echo [8/12] Creating IAM trust policy file...
(
echo {
echo   "Version": "2012-10-17",
echo   "Statement": [
echo     {
echo       "Effect": "Allow",
echo       "Principal": {
echo         "Service": "ecs-tasks.amazonaws.com"
echo       },
echo       "Action": "sts:AssumeRole"
echo     }
echo   ]
echo }
) > ecs-task-trust-policy.json

copy /Y ecs-task-trust-policy.json ecs-task-execution-trust.json >nul

REM ===== Execution role =====
echo.
echo [9/12] Ensuring ECS execution role exists...
aws iam get-role --role-name %EXECUTION_ROLE_NAME% >nul 2>&1
if errorlevel 1 (
    echo Creating %EXECUTION_ROLE_NAME%...
    aws iam create-role ^
      --role-name %EXECUTION_ROLE_NAME% ^
      --assume-role-policy-document file://ecs-task-execution-trust.json
    if errorlevel 1 goto error

    aws iam attach-role-policy ^
      --role-name %EXECUTION_ROLE_NAME% ^
      --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
    if errorlevel 1 goto error
) else (
    echo Execution role already exists.
)

REM ===== Task role =====
echo.
echo [10/12] Ensuring task role with S3 permissions exists...
aws iam get-role --role-name %TASK_ROLE_NAME% >nul 2>&1
if errorlevel 1 (
    echo Creating %TASK_ROLE_NAME%...
    aws iam create-role ^
      --role-name %TASK_ROLE_NAME% ^
      --assume-role-policy-document file://ecs-task-trust-policy.json
    if errorlevel 1 goto error
) else (
    echo Task role already exists.
)

REM ===== S3 policy =====
(
echo {
echo   "Version": "2012-10-17",
echo   "Statement": [
echo     {
echo       "Effect": "Allow",
echo       "Action": [
echo         "s3:PutObject",
echo         "s3:GetObject",
echo         "s3:ListBucket"
echo       ],
echo       "Resource": [
echo         "arn:aws:s3:::%S3_BUCKET%",
echo         "arn:aws:s3:::%S3_BUCKET%/*"
echo       ]
echo     }
echo   ]
echo }
) > s3-guard-trainer-policy.json

aws iam put-role-policy ^
  --role-name %TASK_ROLE_NAME% ^
  --policy-name secureLlmGuardTrainerS3Policy ^
  --policy-document file://s3-guard-trainer-policy.json
if errorlevel 1 goto error

REM ===== Register task definition =====
echo.
echo [11/12] Creating ECS task definition file...

(
echo {
echo   "family": "%TASK_FAMILY%",
echo   "networkMode": "awsvpc",
echo   "requiresCompatibilities": ["FARGATE"],
echo   "cpu": "2048",
echo   "memory": "4096",
echo   "executionRoleArn": "arn:aws:iam::%AWS_ACCOUNT_ID%:role/%EXECUTION_ROLE_NAME%",
echo   "taskRoleArn": "arn:aws:iam::%AWS_ACCOUNT_ID%:role/%TASK_ROLE_NAME%",
echo   "containerDefinitions": [
echo     {
echo       "name": "%CONTAINER_NAME%",
echo       "image": "%ECR_IMAGE%",
echo       "essential": true,
echo       "environment": [
echo         {"name": "USE_S3_UPLOAD", "value": "true"},
echo         {"name": "S3_BUCKET", "value": "%S3_BUCKET%"},
echo         {"name": "S3_PREFIX", "value": "%S3_PREFIX%"},
echo         {"name": "AWS_DEFAULT_REGION", "value": "%AWS_REGION%"}
echo       ],
echo       "logConfiguration": {
echo         "logDriver": "awslogs",
echo         "options": {
echo           "awslogs-group": "%LOG_GROUP%",
echo           "awslogs-region": "%AWS_REGION%",
echo           "awslogs-stream-prefix": "ecs"
echo         }
echo       }
echo     }
echo   ]
echo }
) > ecs-task-def.json

aws ecs register-task-definition ^
  --cli-input-json file://ecs-task-def.json ^
  --region %AWS_REGION%
if errorlevel 1 goto error

REM ===== Get default network =====
echo.
echo [12/12] Resolving default VPC/Subnet/Security Group...

for /f "tokens=*" %%i in ('aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text --region %AWS_REGION%') do set VPC_ID=%%i

for /f "tokens=*" %%i in ('aws ec2 describe-subnets --filters "Name=vpc-id,Values=!VPC_ID!" --query "Subnets[0].SubnetId" --output text --region %AWS_REGION%') do set SUBNET_ID=%%i

for /f "tokens=*" %%i in ('aws ec2 describe-security-groups --filters "Name=vpc-id,Values=!VPC_ID!" "Name=group-name,Values=default" --query "SecurityGroups[0].GroupId" --output text --region %AWS_REGION%') do set SECURITY_GROUP_ID=%%i

echo VPC: !VPC_ID!
echo Subnet: !SUBNET_ID!
echo Security Group: !SECURITY_GROUP_ID!

if "!SUBNET_ID!"=="None" goto network_error
if "!SECURITY_GROUP_ID!"=="None" goto network_error

REM ===== Give IAM a few seconds to propagate =====
echo.
echo Waiting 15 seconds for IAM propagation...
timeout /t 15 /nobreak >nul

REM ===== Run one-time Fargate task =====
echo.
echo Running ECS Fargate one-time training task...

aws ecs run-task ^
  --cluster %ECS_CLUSTER% ^
  --launch-type FARGATE ^
  --task-definition %TASK_FAMILY% ^
  --network-configuration "awsvpcConfiguration={subnets=[!SUBNET_ID!],securityGroups=[!SECURITY_GROUP_ID!],assignPublicIp=ENABLED}" ^
  --region %AWS_REGION% ^
  --query "tasks[0].taskArn" ^
  --output text > latest_task_arn.txt

if errorlevel 1 goto error

set /p TASK_ARN=<latest_task_arn.txt

echo.
echo ================================
echo Task started:
echo !TASK_ARN!
echo ================================

echo.
echo To check task status:
echo aws ecs describe-tasks --cluster %ECS_CLUSTER% --tasks !TASK_ARN! --region %AWS_REGION%

echo.
echo To check logs:
echo aws logs describe-log-streams --log-group-name %LOG_GROUP% --order-by LastEventTime --descending --max-items 1 --region %AWS_REGION%

echo.
echo To check S3 artifacts:
echo aws s3 ls s3://%S3_BUCKET%/%S3_PREFIX%/ --recursive

echo.
echo Done.
goto end

:network_error
echo.
echo ERROR: Could not find default VPC/Subnet/Security Group.
echo You may need to create/select a VPC manually.
goto end

:error
echo.
echo ERROR: Something failed. Check the message above.
goto end

:end
endlocal