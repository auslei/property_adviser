from property_adviser.preprocess_util.preprocess_derive import extract_street

# Comprehensive test cases
addresses = [
    "724-728 Whitehorse Road",
    "123 Main Street",
    "45A/67 Smith Avenue",
    "Unit 4, 89 Johnson Blvd",
    "Flat 12/34 Ocean Drive",
    "Apartment 5B - 123 City Street",
    "12-14A/100 Park Avenue",
    "Lot 5, 20-22 Mountain View",
    "123a Beach Road",
    "Suite 101/500 Business Center",
    "Shop 4, 10 Market Square",
    "Duplex 3/200 Residential Blvd",
    "Studio 2a/15 Arts District",
    "Villa 8/40 Resort Lane",
    "1234 Complex Name, 500 Long Street"
]

expected = [
    "Whitehorse Road",
    "Main Street",
    "Smith Avenue",
    "Johnson Blvd",
    "Ocean Drive",
    "City Street",
    "Park Avenue",
    "Mountain View",
    "Beach Road",
    "Business Center",
    "Market Square",
    "Residential Blvd",
    "Arts District",
    "Resort Lane",
    "Long Street"
]

print("Testing comprehensive street extraction:")
print("----------------------------------------")
for i, (addr, exp) in enumerate(zip(addresses, expected)):
    result = extract_street(addr, {"unknown_value": "Unknown"})
    status = "✓" if result == exp else "✗"
    print(f"{i+1}. Input: '{addr}'")
    print(f"   Expected: '{exp}'")
    print(f"   Result: '{result}' {status}")
    print()

# Calculate and display success rate
success_count = sum(1 for r, e in zip([extract_street(a, {"unknown_value": "Unknown"}) for a in addresses], expected) if r == e)
print(f"Success rate: {success_count}/{len(addresses)} = {success_count/len(addresses)*100:.1f}%")