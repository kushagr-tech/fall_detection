package com.example.activitycollector

import org.json.JSONObject
import java.util.UUID

/**
 * EmergencyContact
 * ================
 * Data model for registered emergency contacts.
 *
 * Contact order matters:
 * - 1st contact (Primary): Receives an automated phone call AND an emergency SMS.
 * - All other contacts: Receive the emergency SMS with GPS location.
 */
data class EmergencyContact(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val phone: String
) {
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("id", id)
            put("name", name)
            put("phone", phone)
        }
    }

    companion object {
        fun fromJson(json: JSONObject): EmergencyContact {
            return EmergencyContact(
                id = json.optString("id", UUID.randomUUID().toString()),
                name = json.getString("name"),
                phone = json.getString("phone")
            )
        }
    }
}
