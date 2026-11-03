# VPN Setup Guide

Owner: IT Operations. Remote access to internal Northwind systems requires the corporate VPN, which runs on WireGuard and is managed through the NorthGate client.

## Installing the client

1. Open Self Service on macOS or Software Center on Windows and install NorthGate.
2. Sign in with your Northwind single sign-on account.
3. Approve the push notification in the Okta Verify app. Hardware security keys also work.

NorthGate downloads your WireGuard profile automatically. Do not copy profiles between devices: each profile is tied to one enrolled laptop.

## Connecting

Choose the gateway closest to you: us-east, eu-west or ap-south. The client picks one automatically if you leave the setting on Auto. Split tunnelling is enabled, so only traffic for internal services goes through the VPN; video calls and general browsing do not.

## Troubleshooting

- If the client shows "handshake did not complete", check that UDP port 51820 is not blocked by your local network. Hotel and airport networks often block it; switch to the TCP fallback in Settings.
- If you see the error NW-4012, your device certificate has expired. Re-enroll the laptop from Self Service to get a new certificate.
- If internal hostnames do not resolve, disconnect, run the "Reset DNS" action in NorthGate, and reconnect.

## Access on personal devices

Personal phones and tablets cannot connect to the VPN. Use the web versions of email and chat on personal devices instead. Contractors receive VPN access only for the duration of their contract; it is removed automatically on the end date.

For further help, open a ticket in the IT Help portal under the category "Remote Access".
