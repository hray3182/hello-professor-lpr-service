import stun
import socket # Added for more specific error catching

# List of STUN servers to test
stun_servers_to_test = [
    ('stun.l.google.com', 19302),
    ('stun1.l.google.com', 19302),
    ('stun.xten.com', 3478),
    ('stunserver.org', 3478),
    ('stun.ekiga.net', 3478) # Add a few more from your list if needed
]

print("Attempting to test STUN server connectivity...\n")

for server_host, server_port in stun_servers_to_test:
    print(f"--- Testing STUN server: {server_host}:{server_port} ---")
    try:
        # stun.get_ip_info can sometimes hang if the server is unresponsive or UDP is blocked.
        # You might need to adjust source_ip and source_port if you have multiple network interfaces
        # or specific firewall rules. For most cases, default is fine.
        # Default timeout for the socket used by pystun3 is 1 second for recv.
        nat_type, external_ip, external_port = stun.get_ip_info(
            stun_host=server_host,
            stun_port=server_port,
            source_ip='0.0.0.0', # Bind to all available interfaces
            source_port=0      # Let OS choose a source port
        )
        print(f"  NAT Type: {nat_type}")
        print(f"  External IP: {external_ip}")
        print(f"  External Port: {external_port}")

        if external_ip and external_ip != '0.0.0.0' and not external_ip.startswith('192.168.') and not external_ip.startswith('10.') and not external_ip.startswith('172.16.'): # Basic check for public IP
            print(f"  STUN request to {server_host}:{server_port} likely SUCCEEDED in getting a public IP.\n")
        elif external_ip:
            print(f"  STUN request to {server_host}:{server_port} returned an IP ({external_ip}), but it might be a local/private IP. Please verify.\n")
        else:
            print(f"  STUN request to {server_host}:{server_port} FAILED to retrieve an external IP (got None or empty).\n")

    except stun.StunException as e:
        # StunException can be raised for various reasons, including timeout if no response.
        print(f"  STUN error for {server_host}:{server_port}: {e}\n")
    except socket.gaierror as e:
        print(f"  DNS resolution error for {server_host}: {e}. Check server name or your DNS settings.\n")
    except socket.error as e:
        # This can catch more generic socket errors, e.g., "Network is unreachable" or "Connection refused"
        # although for UDP, "Connection refused" is less common from the server itself.
        # More likely to be a local OS issue preventing binding or sending.
        print(f"  Socket error for {server_host}:{server_port}: {e}\n")
    except Exception as e:
        print(f"  An unexpected error occurred with {server_host}:{server_port}: {e}\n")

print("--- STUN Test Complete ---")
print("Interpretation Help:")
print("- If you see an 'External IP' that is your public IP address, that STUN server is working for this machine.")
print("- If 'External IP' is None, empty, a private IP (e.g., 192.168.x.x), or you see errors like 'StunException: Server attribution failed', 'timed out', 'Network is unreachable', it means this machine could NOT successfully use that STUN server.")
print("- This test checks if your server can *reach out* to STUN servers. It doesn't guarantee WebRTC will fully work, but it's a critical first step for your aiortc service.")
