
## vpc 생성 ##

```
export AWS_REGION="ap-northeast-2"
export KEYPAIR_NAME="no-key"

cd ~
git clone https://github.com/gnosia93/vlm-distillation.git
cd ~/vlm-distillation
pwd

MY_IP="$(curl -s https://checkip.amazonaws.com)""/32"
echo ${MY_IP}
sed -i "" "s|\${MY_IP}|$MY_IP|g" $(pwd)/cf/eks-vpc.yaml

aws cloudformation create-stack \
  --region ${AWS_REGION} \
  --stack-name get-started-eks \
  --template-body file://$(pwd)/cf/eks-vpc.yaml \
  --parameters ParameterKey=KeyPairName,ParameterValue=${KEYPAIR_NAME} \
  --capabilities CAPABILITY_NAMED_IAM \
  --tags Key=Project,Value=get-started-eks
```
vpc 생성 진행 과정을 조회하고 완료될때 까지 대기한다. 
```
aws cloudformation describe-stacks --stack-name get-started-eks --query "Stacks[0].StackStatus"
```

생성 결과를 출력한다. 
```
OUTPUT=$(aws cloudformation describe-stacks --region ${AWS_REGION} \
  --stack-name get-started-eks \
  --query "Stacks[0].Outputs[?OutputKey=='GravitonVsCode' || OutputKey=='X86VsCode'].\
  {Name: OutputKey, Value: OutputValue}" \
  --output table)
echo ${OUTPUT}
```
[결과]
```
-------------------------------------------------------------------------------------------
|                                     DescribeStacks                                      |
+----------------+------------------------------------------------------------------------+
|      Name      |                                 Value                                  |
+----------------+------------------------------------------------------------------------+
|  GravitonVsCode|  http://ec2-43-200-252-173.ap-northeast-2.compute.amazonaws.com:9090   |
|  X86VsCode     |  http://ec2-54-180-202-252.ap-northeast-2.compute.amazonaws.com:9090   |
+----------------+------------------------------------------------------------------------+
```

## vpc 삭제하기 ##
```
aws cloudformation delete-stack --stack-name get-started-eks
```
