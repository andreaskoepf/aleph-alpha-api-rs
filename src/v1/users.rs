use serde::{Deserialize, Serialize};

// // custom serde field deserialization (could be to handle credits_remaining & out_of_credits_threshold)
// pub fn parse_string_into<'a, D, T>(d: D) -> Result<T, D::Error>
// where
//     D: Deserializer<'a>,
//     T: std::str::FromStr,
// {
//     use serde::de::Error;

//     let val = String::deserialize(d)?;
//     let v = val
//         .parse::<T>()
//         .map_err(|_| Error::custom("failed to parse field value"))?;
//     Ok(v)
// }

#[derive(Deserialize, Debug)]
pub struct UserDetail {
    /// User ID
    pub id: i64,
    /// Email address of the user
    pub email: String,
    /// Role of the user
    pub role: String,
    /// Remaining credits for this user
    pub credits_remaining: String, // (Note: API 1.13.0 returns value as string)
    /// Is this user post-paid?
    pub invoice_allowed: bool,
    /// Threshold for out-of-credits notification. If the threshold gets crossed with a task, then we trigger an email.
    pub out_of_credits_threshold: String, // (Note: API 1.13.0 returns value as string)
    /// Version string of the terms of service that the user has accepted
    pub terms_of_service_version: String,
}

#[derive(Serialize, Debug)]
pub struct UserChange {
    /// Threshold for out-of-credits notification. If the threshold gets crossed with a task, then we trigger an email.
    out_of_credits_threshold: i32,
}
