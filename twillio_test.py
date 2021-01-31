#twillio test
# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client

text_sent_sd = False
# Your Account Sid and Auth Token from twilio.com/console
# and set the environment variables. See http://twil.io/secure
#account_sid = os.environ['AC3f9cf700ee5479b844ca208a38252005']
#auth_token = os.environ['fb3b26b83828e0477d261079cafc3562']
def send_message(message,phone_number):
    account_sid = "AC3f9cf700ee5479b844ca208a38252005" #os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = "c0d462e711b0bc282380102b20c5b886" #os.environ.get('TWILIO_AUTH_TOKEN')
    client = Client(account_sid, auth_token)

    message = client.messages \
                    .create(
                        body=message,
                        from_='+19495652969',
                        to=phone_number
                    )

    print(message.sid)

