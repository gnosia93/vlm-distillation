
```
export REGION=ap-northeast-2
export AZ=ap-northeast-2a

VPC_ID=$(aws ec2 create-vpc \
  --region $REGION \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=vlm-vpc}]' \
  --query 'Vpc.VpcId' --output text)

echo "VPC_ID=$VPC_ID"

# DNS 이름 해석 활성화 (퍼블릭 DNS 붙으려면 필요)
aws ec2 modify-vpc-attribute --region $REGION --vpc-id $VPC_ID --enable-dns-support
aws ec2 modify-vpc-attribute --region $REGION --vpc-id $VPC_ID --enable-dns-hostnames
```
